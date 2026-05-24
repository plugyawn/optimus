from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def population_from_run(run_dir: Path, summary: dict[str, Any]) -> int:
    if summary.get("population") is not None:
        return int(summary["population"])
    name = run_dir.name
    if name.startswith("search_p"):
        return int(name.split("_", 2)[1][1:])
    raise ValueError(f"could not infer population for {run_dir}")


def selected_candidate(summary: dict[str, Any], selection: str) -> str:
    rows = summary.get(selection) or []
    if not rows:
        raise ValueError(f"summary has no non-empty {selection!r} selection row")
    candidate = rows[0].get("candidate")
    if not candidate:
        raise ValueError(f"first {selection!r} row has no candidate field")
    return str(candidate)


def resolve_adapter_path(run_dir: Path, adapter_row: dict[str, Any]) -> Path:
    recorded = Path(str(adapter_row["path"]))
    if recorded.exists():
        return recorded
    fallback = run_dir / "adapters" / recorded.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"adapter path not found: recorded={recorded} fallback={fallback}")


def selected_adapter(run_dir: Path, selection: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    summary = read_json(run_dir / "summary.json")
    candidate = selected_candidate(summary, selection)
    adapters = read_jsonl(run_dir / "adapters.jsonl")
    by_candidate = {str(row.get("candidate")): row for row in adapters}
    adapter = by_candidate.get(candidate)
    if adapter is None:
        raise ValueError(f"selected candidate {candidate!r} not found in {run_dir / 'adapters.jsonl'}")
    return summary, adapter, resolve_adapter_path(run_dir, adapter)


def copy_adapter(adapter_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(adapter_path, output_dir)


def merge_adapter(base_model: str, adapter_path: Path, output_dir: Path, *, torch_dtype: str, device_map: str, max_shard_size: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[torch_dtype]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    del merged, peft_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_run(run_dir: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary, adapter, adapter_path = selected_adapter(run_dir, args.selection)
    population = population_from_run(run_dir, summary)
    output_dir = output_root / f"p{population}"
    if args.mode == "adapter":
        copy_adapter(adapter_path, output_dir)
    else:
        merge_adapter(
            args.model or str(summary["model"]),
            adapter_path,
            output_dir,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_shard_size=args.max_shard_size,
        )
    return {
        "population": population,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "model": args.model or summary.get("model"),
        "selection": args.selection,
        "candidate": adapter["candidate"],
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export selected Optimus adapters as adapter dirs or merged model dirs.")
    parser.add_argument("--run", action="append", default=[], help="Search run directory. Can be repeated.")
    parser.add_argument("--root", type=Path, help="Search root containing search_p*_chunk* directories.")
    parser.add_argument("--out-root", type=Path, default=Path("results/materialized"))
    parser.add_argument("--selection", default="top_screen", choices=["top_screen", "top_holdout"])
    parser.add_argument("--mode", default="merged", choices=["adapter", "merged"])
    parser.add_argument("--model", default="", help="Override base model for merged export.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--manifest-out", type=Path)
    return parser


def run_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    dirs = [Path(item) for item in args.run]
    if args.root:
        dirs.extend(sorted(args.root.glob("search_p*_chunk*")))
    unique = []
    seen = set()
    for path in dirs:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "summary.json").exists() and (path / "adapters.jsonl").exists():
            unique.append(path)
    if not unique:
        raise ValueError("no search run directories with summary.json and adapters.jsonl were found")
    return unique


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [materialize_run(run_dir, args.out_root, args) for run_dir in run_dirs_from_args(args)]
    rows = sorted(rows, key=lambda row: int(row["population"]))
    manifest = {"kind": "optimus_selected_adapter_materialization", "rows": rows}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_out = args.manifest_out or args.out_root / "manifest.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
