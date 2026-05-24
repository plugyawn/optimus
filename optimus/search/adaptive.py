from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch

from optimus.core.perturbations import PerturbationSpec as Candidate, parse_perturbation_key, perturbation_panel
from optimus.defaults import DEFAULT_MODEL, DEFAULT_TARGETS
from optimus.modeling.noise import lora_noise_tensors
from optimus.search.ensemble import anzo_anchor_prompts
from optimus.search.peft import evaluate_candidate, make_backend, reset_outputs, tag_rows, write_jsonl
from optimus.tasks.countdown import load_examples, prompts as make_prompts, unique_example_count, unique_semantic_example_count


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_prior_paths(text: str) -> list[Path]:
    paths: list[Path] = []
    for item in (x.strip() for x in text.split(",") if x.strip()):
        matches = glob.glob(item)
        paths.extend(Path(x) for x in (matches or [item]))
    return paths


def candidate_score_rows(paths: list[Path], top_k: int, min_score: float) -> list[dict]:
    rows = []
    for path in paths:
        files = [path] if path.is_file() else [
            path / "candidate_summary.jsonl",
            path / "stage_candidate_summary.jsonl",
            path / "summary.json",
        ]
        for file in files:
            if not file.exists():
                continue
            if file.name == "summary.json":
                summary = json.loads(file.read_text())
                for section in ("top_screen", "top_holdout", "top_stage"):
                    for row in summary.get(section, []) or []:
                        row = dict(row)
                        row["source"] = str(file)
                        rows.append(row)
            else:
                for row in read_jsonl(file):
                    row = dict(row)
                    row["source"] = str(file)
                    rows.append(row)

    best_by_key: dict[str, dict] = {}
    for row in rows:
        key = row.get("candidate")
        if not key or key == "base":
            continue
        score = float(row.get("exact_mean", row.get("stage_exact_mean", 0.0)))
        if score < min_score:
            continue
        try:
            parse_perturbation_key(key)
        except Exception:
            continue
        prev = best_by_key.get(key)
        if prev is None or score > float(prev.get("exact_mean", prev.get("stage_exact_mean", 0.0))):
            row["score_for_basis"] = score
            best_by_key[key] = row
    return sorted(best_by_key.values(), key=lambda r: r["score_for_basis"], reverse=True)[:top_k]


def lora_a_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and module.lora_A:
            adapter = next(iter(module.lora_A.keys()))
            yield name, module.lora_A[adapter].weight


def candidate_a_noise(name: str, weight: torch.Tensor, candidate: Candidate) -> torch.Tensor:
    unit_candidate = Candidate(candidate.family, candidate.seed, 1.0, candidate.sign, method=candidate.method)
    noise, _ = lora_noise_tensors(name, tuple(weight.shape), (1, 1), unit_candidate, rank=max(1, weight.shape[0]))
    return noise.detach().cpu()


def top_basis(rows: list[torch.Tensor], basis_rank: int) -> torch.Tensor | None:
    rows = [x.float().cpu() for x in rows if x.numel() > 0]
    if not rows:
        return None
    x = torch.cat(rows, dim=0)
    if x.shape[0] == 0:
        return None
    x = x - x.mean(dim=0, keepdim=True)
    if x.abs().sum().item() == 0.0:
        return None
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[: min(basis_rank, vh.shape[0])].contiguous()


def col_scale_from_rows(rows: list[torch.Tensor], strength: float, clamp: tuple[float, float]) -> torch.Tensor | None:
    rows = [x.float().cpu() for x in rows if x.numel() > 0]
    if not rows:
        return None
    x = torch.cat(rows, dim=0)
    var = x.pow(2).mean(dim=0)
    scale = torch.sqrt(var / var.mean().clamp_min(1e-12))
    scale = 1.0 + strength * (scale - 1.0)
    return scale.clamp(clamp[0], clamp[1]).contiguous()


def build_family_state(args, backend, screen, prior_rows: list[dict], current_rows: list[dict]) -> tuple[dict, list[dict]]:
    sources = {x.strip() for x in args.basis_source.split(",") if x.strip()}
    activation_state = {}
    if "activation" in sources or "hybrid" in sources:
        target = make_prompts(screen[: min(args.activation_prompts, len(screen))])
        activation_state = backend.build_anzo_state(target, anzo_anchor_prompts())

    scored_rows = []
    if "elite" in sources or "hybrid" in sources:
        scored_rows.extend(prior_rows)
    if "current" in sources:
        scored_rows.extend(sorted(current_rows, key=lambda r: r["exact_mean"], reverse=True)[: args.basis_elites])

    state = {}
    summary = []
    basis_rank = max(1, args.basis_rank)
    for name, weight in lora_a_modules(backend.model):
        basis_inputs: list[torch.Tensor] = []
        cov_inputs: list[torch.Tensor] = []
        source_counts = {"activation": 0, "elite": 0}

        if name in activation_state:
            act = activation_state[name].detach().cpu().float()
            basis_inputs.append(act)
            cov_inputs.append(act)
            source_counts["activation"] += int(act.shape[0])

        for row in scored_rows[: args.basis_elites]:
            cand = parse_perturbation_key(row["candidate"])
            if cand.family == "anzo" and name in activation_state:
                noise = cand.sign * activation_state[name].detach().cpu().float()
            else:
                noise = candidate_a_noise(name, weight, cand)
            score = max(float(row.get("score_for_basis", row.get("exact_mean", 0.0))), args.min_basis_weight)
            signed = noise * math.sqrt(score)
            basis_inputs.append(signed)
            cov_inputs.append(signed)
            source_counts["elite"] += int(signed.shape[0])

        basis = top_basis(basis_inputs, basis_rank)
        col_scale = col_scale_from_rows(cov_inputs, args.cov_strength, (args.col_scale_min, args.col_scale_max))
        if basis is None and col_scale is None:
            continue
        spec = {
            "mode": "activation_overwrite" if args.mode == "activation-overwrite" else args.mode.replace("-", "_"),
            "basis_scale": args.basis_scale,
            "residual_scale": args.residual_scale,
        }
        if basis is not None:
            spec["basis"] = basis
        if col_scale is not None and args.mode in {"covlite", "hybrid-covlite"}:
            spec["col_scale"] = col_scale
        state[name] = spec
        summary.append(
            {
                "module": name,
                "basis_rows": int(0 if basis is None else basis.shape[0]),
                "in_features": int(weight.shape[1]),
                "activation_rows": source_counts["activation"],
                "elite_rows": source_counts["elite"],
                "col_scale_min": None if col_scale is None else float(col_scale.min().item()),
                "col_scale_max": None if col_scale is None else float(col_scale.max().item()),
                "col_scale_mean": None if col_scale is None else float(col_scale.mean().item()),
            }
        )
    return state, summary


def jsonable_eval(ev: dict, **extra) -> dict:
    row = {k: v for k, v in ev.items() if k != "rows"}
    row.update(extra)
    return row


def run_search(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(
        out,
        [
            "per_prompt.jsonl",
            "candidate_summary.jsonl",
            "holdout_per_prompt.jsonl",
            "basis_summary.json",
        ],
    )
    backend = make_backend(args)
    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    prior_rows = candidate_score_rows(expand_prior_paths(args.prior_results), args.basis_elites, args.min_prior_score)

    base_screen = evaluate_candidate(backend, None, screen, args)
    base_holdout = evaluate_candidate(backend, None, holdout, args)
    write_jsonl(out / "per_prompt.jsonl", tag_rows(base_screen["rows"], mode="base_screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(base_holdout["rows"], mode="base_holdout"))

    all_rows: list[dict] = []
    round_states: list[dict] = []
    round_basis_summaries: list[dict] = []
    start = time.time()
    state, basis_summary = build_family_state(args, backend, screen, prior_rows, all_rows)

    for round_idx in range(args.rounds):
        round_states.append(state)
        torch.save(state, out / f"family_state_round{round_idx}.pt")
        round_basis_summaries.append({"round": round_idx, "modules": basis_summary})
        family = f"adaptive_{args.mode.replace('-', '_')}_r{round_idx}"
        candidates = perturbation_panel(
            "lora",
            family,
            args.population,
            args.sigma,
            args.seed + round_idx * 1_000_003,
            args.antithetic,
        )
        for i, cand in enumerate(candidates):
            ev = evaluate_candidate(backend, cand, screen, args, state)
            row = jsonable_eval(ev, round=round_idx, mode=args.mode)
            all_rows.append(row)
            prompt_rows = [dict(x, round=round_idx, mode="screen", family_mode=args.mode) for x in ev["rows"]]
            write_jsonl(out / "per_prompt.jsonl", prompt_rows)
            print(
                f"round={round_idx} {i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f} "
                f"malformed={ev['malformed_mean']:.4f}",
                flush=True,
            )
        if round_idx + 1 < args.rounds:
            state, basis_summary = build_family_state(args, backend, screen, prior_rows, all_rows)

    top = sorted(all_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(all_rows))]
    holdout_rows = []
    for row in top:
        state = round_states[int(row["round"])]
        ev = evaluate_candidate(backend, parse_perturbation_key(row["candidate"]), holdout, args, state)
        holdout_row = jsonable_eval(ev, round=int(row["round"]), mode=args.mode, screen_exact_mean=row["exact_mean"])
        holdout_rows.append(holdout_row)
        write_jsonl(
            out / "holdout_per_prompt.jsonl",
            [dict(x, round=int(row["round"]), mode="holdout", family_mode=args.mode) for x in ev["rows"]],
        )

    total_s = time.time() - start
    best_holdout = max((x["exact_mean"] for x in holdout_rows), default=None)
    summary = {
        "kind": "adaptive_search",
        "mode": args.mode,
        "basis_source": args.basis_source,
        "prior_results": args.prior_results,
        "prior_elites_loaded": len(prior_rows),
        "rounds": args.rounds,
        "population_per_round": args.population,
        "population_total": len(all_rows),
        "sigma": args.sigma,
        "antithetic": args.antithetic,
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "best_holdout_exact": best_holdout,
        "best_holdout_delta": None if best_holdout is None else best_holdout - base_holdout["exact_mean"],
        "no_quality_regression": None if best_holdout is None else best_holdout >= base_holdout["exact_mean"],
        "candidate_sec": len(all_rows) / max(total_s, 1e-9),
        "prompt_eval_sec": (len(all_rows) * len(screen) + len(top) * len(holdout)) / max(total_s, 1e-9),
        "top_screen": top,
        "top_holdout": holdout_rows,
        "basis_rounds": round_basis_summaries,
        "prior_elites": [
            {
                "candidate": row["candidate"],
                "score_for_basis": row["score_for_basis"],
                "source": row.get("source"),
            }
            for row in prior_rows
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "candidate_summary.jsonl", all_rows)
    (out / "basis_summary.json").write_text(json.dumps(round_basis_summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def add_common_args(sp):
    sp.add_argument("--out", required=True)
    sp.add_argument("--model", default=DEFAULT_MODEL)
    sp.add_argument("--data", default=None)
    sp.add_argument("--prompts", type=int, default=32)
    sp.add_argument("--holdout-prompts", type=int, default=32)
    sp.add_argument("--seed", type=int, default=1234)
    sp.add_argument("--rank", type=int, default=8)
    sp.add_argument("--sigma", type=float, default=0.02)
    sp.add_argument("--targets", default=DEFAULT_TARGETS)
    sp.add_argument("--max-new-tokens", type=int, default=32)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    sp.add_argument("--prompt-variant", default="default")
    sp.add_argument("--use-chat-template", action="store_true")
    sp.add_argument("--stop-at-answer", action="store_true")
    sp.add_argument("--allow-repeat-data", action="store_true")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Adaptive LoRA perturbation-basis experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search", help="Run elite-basis or covariance-lite adaptive perturbation search.")
    add_common_args(sp)
    sp.add_argument("--mode", choices=["elite-basis", "covlite", "hybrid-covlite", "activation-overwrite"], default="hybrid-covlite")
    sp.add_argument("--basis-source", default="hybrid,current", help="Comma list: activation, elite, current, hybrid.")
    sp.add_argument("--prior-results", default="", help="Comma-separated result dirs/files/globs with candidate summaries.")
    sp.add_argument("--population", type=int, default=64)
    sp.add_argument("--rounds", type=int, default=2)
    sp.add_argument("--promote", type=int, default=4)
    sp.add_argument("--basis-elites", type=int, default=16)
    sp.add_argument("--basis-rank", type=int, default=16)
    sp.add_argument("--activation-prompts", type=int, default=16)
    sp.add_argument("--basis-scale", type=float, default=1.0)
    sp.add_argument("--residual-scale", type=float, default=0.5)
    sp.add_argument("--cov-strength", type=float, default=0.75)
    sp.add_argument("--col-scale-min", type=float, default=0.25)
    sp.add_argument("--col-scale-max", type=float, default=4.0)
    sp.add_argument("--min-prior-score", type=float, default=0.0)
    sp.add_argument("--min-basis-weight", type=float, default=0.05)
    sp.add_argument("--antithetic", action="store_true")
    sp.set_defaults(perturbation_backend="lora")
    args = p.parse_args(argv)
    if args.cmd == "search":
        run_search(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
