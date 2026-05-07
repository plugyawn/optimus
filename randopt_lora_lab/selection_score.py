from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def parse_prompt_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    return variants or ["default"]


def condition_score(
    candidate: dict,
    base: dict,
    *,
    malformed_penalty: float,
    cap_hit_penalty: float,
) -> dict:
    exact_lift = float(candidate["exact_mean"]) - float(base["exact_mean"])
    malformed_regression = float(candidate["malformed_mean"]) - float(base["malformed_mean"])
    cap_hit_regression = float(candidate["cap_hit_mean"]) - float(base["cap_hit_mean"])
    selection_score = (
        exact_lift
        - malformed_penalty * max(malformed_regression, 0.0)
        - cap_hit_penalty * max(cap_hit_regression, 0.0)
    )
    return {
        "base_exact_mean": float(base["exact_mean"]),
        "base_malformed_mean": float(base["malformed_mean"]),
        "base_cap_hit_mean": float(base["cap_hit_mean"]),
        "exact_lift_vs_base": exact_lift,
        "malformed_regression_vs_base": malformed_regression,
        "cap_hit_regression_vs_base": cap_hit_regression,
        "condition_selection_score": selection_score,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def combine_candidate_conditions(
    condition_rows: list[dict],
    base_by_variant: dict[str, dict],
    *,
    score_mode: str,
    malformed_penalty: float,
    cap_hit_penalty: float,
) -> list[dict]:
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    enriched_conditions = []
    for row in condition_rows:
        variant = str(row.get("prompt_variant", "default"))
        base = base_by_variant[variant]
        enriched = dict(
            row,
            **condition_score(
                row,
                base,
                malformed_penalty=malformed_penalty,
                cap_hit_penalty=cap_hit_penalty,
            ),
        )
        enriched_conditions.append(enriched)
        by_candidate[str(enriched["candidate"])].append(enriched)

    combined = []
    for candidate, rows in sorted(by_candidate.items()):
        condition_scores = [float(row["condition_selection_score"]) for row in rows]
        exact_lifts = [float(row["exact_lift_vs_base"]) for row in rows]
        malformed_regressions = [float(row["malformed_regression_vs_base"]) for row in rows]
        cap_hit_regressions = [float(row["cap_hit_regression_vs_base"]) for row in rows]
        if score_mode == "exact":
            selection_score = _mean(float(row["exact_mean"]) for row in rows)
        elif score_mode == "robust_mean":
            selection_score = _mean(condition_scores)
        elif score_mode == "robust_min":
            selection_score = min(condition_scores)
        else:
            raise ValueError(f"unknown score_mode: {score_mode}")
        first = rows[0]
        combined.append(
            {
                "candidate": candidate,
                "selection_score": selection_score,
                "score_mode": score_mode,
                "condition_count": len(rows),
                "prompt_variants": sorted({str(row.get("prompt_variant", "default")) for row in rows}),
                "exact_mean": _mean(float(row["exact_mean"]) for row in rows),
                "malformed_mean": _mean(float(row["malformed_mean"]) for row in rows),
                "cap_hit_mean": _mean(float(row["cap_hit_mean"]) for row in rows),
                "answer_closed_mean": _mean(float(row.get("answer_closed_mean", 0.0)) for row in rows),
                "mean_exact_lift_vs_base": _mean(exact_lifts),
                "min_exact_lift_vs_base": min(exact_lifts),
                "mean_condition_selection_score": _mean(condition_scores),
                "min_condition_selection_score": min(condition_scores),
                "max_malformed_regression_vs_base": max(malformed_regressions),
                "max_cap_hit_regression_vs_base": max(cap_hit_regressions),
                "seed": first.get("seed"),
                "sigma": first.get("sigma"),
                "sign": first.get("sign"),
                "adapter_index": first.get("adapter_index"),
                "adapter": first.get("adapter"),
            }
        )
    return combined


def enrich_condition_rows(
    condition_rows: list[dict],
    base_by_variant: dict[str, dict],
    *,
    malformed_penalty: float,
    cap_hit_penalty: float,
) -> list[dict]:
    enriched = []
    for row in condition_rows:
        variant = str(row.get("prompt_variant", "default"))
        enriched.append(
            dict(
                row,
                **condition_score(
                    row,
                    base_by_variant[variant],
                    malformed_penalty=malformed_penalty,
                    cap_hit_penalty=cap_hit_penalty,
                ),
            )
        )
    return enriched
