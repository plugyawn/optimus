from __future__ import annotations

import math
from typing import Any


def _radius_key(radius: float | None, values: dict[str, Any]) -> str:
    if radius is None:
        if not values:
            raise ValueError("scale row has no beta_t_by_radius values")
        return str(next(iter(values)))
    key = f"{float(radius):g}"
    if key not in values:
        raise ValueError(f"summary has no beta for radius {key!r}")
    return key


def _target_site_by_id(state_summary: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in state_summary.get("targets") or []:
        target_id = row.get("target_id")
        site_id = row.get("activation_site_id")
        if target_id is not None and site_id is not None:
            out[str(target_id)] = str(site_id)
    if out:
        return out
    for site in state_summary.get("activation_sites") or []:
        site_id = str(site.get("site_id", ""))
        for target_id in site.get("target_module_ids") or []:
            out[str(target_id)] = site_id
    return out


def _effective_h_s(site: dict[str, Any], effective_rank: int | None) -> float:
    source_h = float(site["H_s"])
    if effective_rank is None:
        return source_h
    source_rank = int(site.get("effective_rank") or site.get("requested_rank") or 0)
    rank = int(effective_rank)
    if rank <= 0:
        raise ValueError(f"effective_rank must be positive for {site.get('site_id')}, got {rank}")
    if rank > source_rank:
        raise ValueError(f"effective_rank must be <= source rank {source_rank} for {site.get('site_id')}, got {rank}")
    if rank == source_rank:
        return source_h

    basis_kind = str(site.get("basis_kind") or "")
    singular_values = [float(item) for item in (site.get("singular_values") or [])]
    tokens = int(site.get("num_calibration_tokens") or 0)
    if basis_kind not in {"activation-svd", "shuffled-activation-svd"}:
        raise ValueError(
            "cannot recompute relative-output-rms beta for sliced effective_rank "
            f"{rank} from basis_kind={basis_kind!r}; rerun search with --basis-rank {rank}"
        )
    if tokens <= 0 or len(singular_values) < rank:
        raise ValueError(
            "cannot recompute relative-output-rms beta for sliced effective_rank "
            f"{rank}: missing singular values or calibration token count for {site.get('site_id')}"
        )
    h_s = sum(value * value for value in singular_values[:rank]) / float(tokens)
    if not math.isfinite(h_s) or h_s <= 0.0:
        raise ValueError(f"recomputed H_s for {site.get('site_id')} is not positive finite: {h_s}")
    return h_s


def resolved_betas_by_target(
    summary: dict[str, Any],
    *,
    radius: float | None = None,
    scale_multiplier: float = 1.0,
    state_summary: dict[str, Any] | None = None,
    effective_rank: int | None = None,
) -> dict[str, float]:
    """Load beta_t values, recomputing relative-RMS scale for rank-sliced SVD bases.

    Replay scripts may slice a stored basis to a smaller effective rank. For
    ``relative-output-rms`` runs, reusing the source run's beta silently changes
    actual perturbation RMS because ``H_s`` changes with rank. If the source
    artifact contains activation-SVD singular values, beta is corrected by
    ``sqrt(H_source / H_effective)``. For random bases there is no per-rank
    activation energy in the artifact, so this fails closed.
    """

    rows = summary.get("resolved_target_scales") or []
    if not rows:
        raise ValueError("source summary is missing resolved_target_scales")
    scale_mode = str(summary.get("scale_mode") or "")
    needs_rank_adjust = effective_rank is not None and scale_mode == "relative-output-rms"

    site_by_id: dict[str, dict[str, Any]] = {}
    target_site: dict[str, str] = {}
    if needs_rank_adjust:
        if state_summary is None:
            raise ValueError("state_summary is required when slicing relative-output-rms effective_rank")
        site_by_id = {str(site["site_id"]): site for site in state_summary.get("activation_sites") or []}
        target_site = _target_site_by_id(state_summary)
        if not site_by_id or not target_site:
            raise ValueError("state_summary is missing activation-site/target metadata for effective_rank beta recompute")

    out: dict[str, float] = {}
    for row in rows:
        target_id = str(row["target_id"])
        values = row.get("beta_t_by_radius") or {}
        key = _radius_key(radius, values)
        beta = float(values[key])
        if needs_rank_adjust:
            site_id = target_site.get(target_id)
            if site_id is None and target_id.endswith(".self_attn.qkv_proj"):
                prefix = target_id[: -len("qkv_proj")]
                site_id = target_site.get(prefix + "q_proj") or target_site.get(prefix + "v_proj")
            if site_id is None or site_id not in site_by_id:
                raise ValueError(f"cannot map target {target_id!r} to an activation site for beta recompute")
            site = site_by_id[site_id]
            source_h = float(site["H_s"])
            effective_h = _effective_h_s(site, effective_rank)
            beta *= math.sqrt(source_h / effective_h)
        out[target_id] = float(scale_multiplier) * beta
    return out
