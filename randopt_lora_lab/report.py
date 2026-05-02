from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in root.glob("*/summary.json"):
        row = json.loads(path.read_text())
        row["run"] = path.parent.name
        summaries.append(row)
    flat = []
    for row in summaries:
        flat.append(
            {
                "run": row["run"],
                "kind": row.get("kind"),
                "family": row.get("family", ""),
                "population": row.get("population", 0),
                "base_screen_exact": row.get("base_screen_exact", row.get("base_exact")),
                "candidate_sec": row.get("candidate_sec"),
                "pair_sec": row.get("pair_sec"),
            }
        )
    df = pd.DataFrame(flat)
    df.to_csv(out / "summary.csv", index=False)
    if "candidate_sec" in df and df["candidate_sec"].notna().any():
        ax = df.dropna(subset=["candidate_sec"]).plot.bar(x="run", y="candidate_sec", legend=False)
        ax.set_ylabel("candidates/sec")
        plt.tight_layout()
        plt.savefig(out / "candidate_sec.png", dpi=160)
        plt.close()
    (out / "report.md").write_text("# RandOpt LoRA Lab Report\n\n" + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
