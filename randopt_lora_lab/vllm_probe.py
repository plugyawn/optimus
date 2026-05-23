from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from optimus.tasks.countdown import load_examples, prompts as make_prompts, score_completion


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_examples(None, args.prompts, args.seed)
    ps = make_prompts(examples)
    load_start = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    load_s = time.time() - load_start
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    gen_start = time.time()
    outputs = llm.generate(ps, sampling, use_tqdm=False)
    elapsed_s = time.time() - gen_start
    rows = []
    output_tokens = 0
    exact = []
    malformed = []
    for ex, item in zip(examples, outputs):
        text = item.outputs[0].text if item.outputs else ""
        token_ids = item.outputs[0].token_ids if item.outputs else []
        output_tokens += len(token_ids)
        score = score_completion(text, ex)
        exact.append(float(score["exact"]))
        malformed.append(float(score["malformed"]))
        rows.append(
            {
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                "output_tokens": len(token_ids),
                **score,
            }
        )
    summary = {
        "kind": "vllm_base_probe",
        "model": args.model,
        "prompts": len(ps),
        "max_new_tokens": args.max_new_tokens,
        "load_s": load_s,
        "elapsed_s": elapsed_s,
        "output_tokens": output_tokens,
        "tokens_per_sec": output_tokens / max(elapsed_s, 1e-9),
        "prompts_per_sec": len(ps) / max(elapsed_s, 1e-9),
        "exact_mean": sum(exact) / len(exact),
        "malformed_mean": sum(malformed) / len(malformed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "per_prompt.jsonl", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
