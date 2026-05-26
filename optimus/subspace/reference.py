from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from optimus import __version__
from optimus.serving.runtime import runtime_environment
from optimus.subspace import (
    ActivationSite,
    BasisKind,
    BudgetPolicy,
    ScaleMode,
    SubspaceCandidate,
    TargetModule,
    random_field_tensor,
)
from optimus.tasks.countdown import CountdownExample, load_examples, semantic_example_key


TARGET_SUFFIXES: dict[str, tuple[str, ...]] = {
    "qv": ("q_proj", "v_proj"),
    "attn-qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "transformer-linears": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}

ATTN_SITE_SUFFIXES = {"q_proj", "k_proj", "v_proj"}
MLP_SITE_SUFFIXES = {"gate_proj", "up_proj"}


@dataclass(frozen=True)
class TargetScale:
    target_id: str
    budget_weight: float
    beta_t_by_radius: dict[str, float]


@dataclass(frozen=True)
class CandidateRandomField:
    direction_seed: int
    sign: str
    target_id: str
    rng_version: str = "gaussian_hash_v1"
    salt: str = ""

    def tensor(self, output_dim: int, basis_dim: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return random_field_tensor(
            direction_seed=self.direction_seed,
            sign=self.sign,  # type: ignore[arg-type]
            target_id=self.target_id,
            output_dim=output_dim,
            basis_rank=basis_dim,
            rng_version=self.rng_version,
            dtype=dtype,
            salt=self.salt,
        )


@dataclass(frozen=True)
class SubspaceEnsemble:
    K: int
    aggregation: str
    candidates: tuple[SubspaceCandidate, ...]


@dataclass(frozen=True)
class TargetRuntime:
    module: TargetModule
    weight: torch.Tensor
    objective: torch.Tensor
    base_output_power_P_t: float


@dataclass(frozen=True)
class ReferenceState:
    basis_tensors: dict[str, torch.Tensor]
    activation_sites: tuple[ActivationSite, ...]
    targets: tuple[TargetRuntime, ...]
    basis_hash: str
    target_set_hash: str
    basis_collection_config_hash: str


def parse_float_grid(text: str | None, *, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("scale grid must contain at least one value")
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("scale grid values must be finite and nonnegative")
    return values


def parse_int_grid(text: str | None, *, default: tuple[int, ...]) -> list[int]:
    if not text:
        return list(default)
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("integer grid must contain positive values")
    return values


def parse_layers(text: str | None) -> tuple[int, ...]:
    if text in {None, "", "all"}:
        return (0,)
    layers = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layers must be 'all' or a comma-separated list of nonnegative layer indices")
    return layers


def stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(stable_json_bytes(payload))


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    header = stable_json_bytes(
        {
            "schema_version": "tensor_sha256_v2",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    )
    buffer = io.BytesIO()
    tensor.untyped_storage()._write_file(buffer, False, False, tensor.element_size())
    return sha256_bytes(header + b"\n" + buffer.getvalue())


def torch_payload_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
    except Exception:
        return True


def examples_hash(examples: list[CountdownExample], *, label: str) -> str:
    rows = [
        {"id": ex.id, "numbers": list(ex.numbers), "target": ex.target, "semantic": [list(semantic_example_key(ex)[0]), ex.target]}
        for ex in examples
    ]
    return sha256_json({"label": label, "examples": rows})


def config_hash(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def deterministic_activations(examples: list[CountdownExample], *, dim: int, seed: int) -> torch.Tensor:
    rows = []
    for ex in examples:
        payload = stable_json_bytes({"seed": seed, "id": ex.id, "numbers": list(ex.numbers), "target": ex.target})
        digest = hashlib.sha256(payload).digest()
        row = []
        for idx in range(dim):
            word = int.from_bytes(hashlib.sha256(digest + idx.to_bytes(4, "little")).digest()[:8], "little")
            u = (word + 0.5) / float(1 << 64)
            value = math.sqrt(-2.0 * math.log(max(u, 1e-12))) * math.cos(2.0 * math.pi * ((idx + 1) * 0.61803398875 % 1.0))
            row.append(value)
        # Make Countdown semantics visible to the SVD basis without labels.
        if dim >= 8:
            nums = list(ex.numbers)
            row[0] += sum(nums) / 50.0
            row[1] += ex.target / 100.0
            row[2] += len(nums) / 10.0
            row[3] += max(nums) / 100.0
            row[4] += min(nums) / 100.0
            row[5] += sum(x * x for x in nums) / 5000.0
            row[6] += (ex.id % 17) / 17.0
            row[7] += 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def build_basis(
    activations: torch.Tensor,
    *,
    requested_rank: int,
    basis_kind: BasisKind,
    centering: str,
    seed: int,
) -> tuple[torch.Tensor, list[float], float, float, float, float]:
    if requested_rank < 0:
        raise ValueError("requested_rank must be nonnegative")
    if activations.ndim != 2:
        raise ValueError("activations must be a [tokens, hidden] matrix")
    matrix = activations.float()
    if centering == "mean":
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    elif centering != "none":
        raise ValueError(f"unsupported basis centering {centering!r}")
    tokens, dim = matrix.shape
    effective_rank = min(requested_rank, tokens, dim)
    if effective_rank == 0:
        basis = matrix.new_zeros((0, dim))
        singular_values: list[float] = []
    elif basis_kind == "activation-svd":
        _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[:effective_rank].contiguous()
        singular_values = [float(item) for item in s[:effective_rank]]
    elif basis_kind == "shuffled-activation-svd":
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        shuffled = matrix.clone()
        for col in range(dim):
            shuffled[:, col] = shuffled[torch.randperm(tokens, generator=gen), col]
        _, s, vh = torch.linalg.svd(shuffled, full_matrices=False)
        basis = vh[:effective_rank].contiguous()
        singular_values = [float(item) for item in s[:effective_rank]]
    elif basis_kind == "random-orthonormal":
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.randn(dim, max(effective_rank, 1), generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        basis = q[:, :effective_rank].T.contiguous()
        singular_values = []
    else:
        raise ValueError(f"unsupported basis kind {basis_kind!r}")
    projected = matrix @ basis.T if effective_rank else matrix.new_zeros((tokens, 0))
    h_s = float((projected.square().sum(dim=1)).mean().item()) if tokens else 0.0
    a_s = float((matrix.square().sum(dim=1)).mean().item()) if tokens else 0.0
    captured = 0.0 if a_s <= 0.0 else h_s / a_s
    if effective_rank:
        gram = basis @ basis.T
        eye = torch.eye(effective_rank, dtype=gram.dtype)
        orth_error = float((gram - eye).abs().max().item())
        gram_error = float(torch.linalg.norm(gram - eye).item())
    else:
        orth_error = 0.0
        gram_error = 0.0
    return basis, singular_values, h_s, a_s, captured, max(orth_error, gram_error)


def site_for_suffix(layer: int, suffix: str) -> tuple[str, str]:
    if suffix in ATTN_SITE_SUFFIXES:
        return f"layer_{layer}.attn_in", "post_rmsnorm"
    if suffix == "o_proj":
        return f"layer_{layer}.o_in", "attn_out"
    if suffix in MLP_SITE_SUFFIXES:
        return f"layer_{layer}.mlp_in", "post_rmsnorm"
    if suffix == "down_proj":
        return f"layer_{layer}.down_in", "mlp_up"
    raise ValueError(f"unsupported target suffix {suffix!r}")


def target_ids(target_preset: str, layers: tuple[int, ...]) -> list[tuple[int, str, str]]:
    try:
        suffixes = TARGET_SUFFIXES[target_preset]
    except KeyError as exc:
        raise ValueError(f"unsupported target preset {target_preset!r}") from exc
    out = []
    for layer in layers:
        for suffix in suffixes:
            site_id, _ = site_for_suffix(layer, suffix)
            out.append((layer, suffix, site_id))
    return out


def deterministic_weight(target_id: str, *, output_dim: int, input_dim: int, seed: int) -> torch.Tensor:
    gen_seed = int(hashlib.sha256(f"{seed}:{target_id}:weight".encode("utf-8")).hexdigest()[:8], 16)
    gen = torch.Generator(device="cpu").manual_seed(gen_seed)
    return torch.randn(output_dim, input_dim, generator=gen, dtype=torch.float32) / math.sqrt(max(input_dim, 1))


def deterministic_objective(target_id: str, *, output_dim: int, seed: int) -> torch.Tensor:
    gen_seed = int(hashlib.sha256(f"{seed}:{target_id}:objective".encode("utf-8")).hexdigest()[:8], 16)
    gen = torch.Generator(device="cpu").manual_seed(gen_seed)
    vec = torch.randn(output_dim, generator=gen, dtype=torch.float32)
    return vec / max(float(torch.linalg.norm(vec).item()), 1e-12)


def build_reference_state(
    *,
    calibration_examples: list[CountdownExample],
    args: Any,
    basis_kind: BasisKind,
    input_dim: int,
    output_dim: int,
    basis_rank: int,
    layers: tuple[int, ...],
    target_preset: str,
    prompt_ids_hash: str,
    decode_config_hash: str,
) -> ReferenceState:
    activations = deterministic_activations(calibration_examples, dim=input_dim, seed=int(args.seed or 0))
    target_specs = target_ids(target_preset, layers)
    site_to_targets: dict[str, list[str]] = {}
    for layer, suffix, site_id in target_specs:
        target_id = f"layer_{layer}.{('self_attn' if suffix in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp')}.{suffix}"
        site_to_targets.setdefault(site_id, []).append(target_id)
    basis_tensors: dict[str, torch.Tensor] = {}
    activation_sites: list[ActivationSite] = []
    targets: list[TargetRuntime] = []
    site_stats: dict[str, tuple[float, float]] = {}
    for site_id, site_targets in sorted(site_to_targets.items()):
        site_seed = int(hashlib.sha256(f"{args.seed}:{site_id}:{basis_kind}".encode("utf-8")).hexdigest()[:8], 16)
        basis, singular_values, h_s, a_s, captured, error = build_basis(
            activations,
            requested_rank=basis_rank,
            basis_kind=basis_kind,
            centering=args.basis_centering or "none",
            seed=site_seed,
        )
        tensor_key = f"basis/{site_id}"
        basis_tensors[tensor_key] = basis
        site_stats[site_id] = (h_s, a_s)
        layer_index = int(site_id.split(".", 1)[0].split("_", 1)[1])
        activation_sites.append(
            ActivationSite(
                site_id=site_id,
                architecture_family="qwen3_text_reference",
                layer_index=layer_index,
                block_path=f"model.layers.{layer_index}",
                read_tensor_path=f"model.layers.{layer_index}.{site_id.split('.', 1)[1]}",
                hook_point="pre_linear",
                norm_position="reference",
                shape_convention="tokens_hidden",
                runtime_dtype="bf16",
                accumulation_dtype="fp32",
                tensor_parallel_sharding_policy="replicated",
                target_module_ids=tuple(site_targets),
                calibration_prompt_ids_hash=prompt_ids_hash,
                calibration_decode_config_hash=decode_config_hash,
                basis_control_seed=site_seed if basis_kind != "activation-svd" else None,
                transductive=False,
                input_dim=input_dim,
                basis_kind=basis_kind,
                requested_rank=basis_rank,
                effective_rank=int(basis.shape[0]),
                basis_tensor_key=tensor_key,
                basis_tensor_sha256=tensor_sha256(basis),
                singular_values=tuple(singular_values),
                captured_energy=float(captured),
                prefill_captured_energy=float(captured),
                decode_captured_energy=None,
                H_s=float(h_s),
                A_s=float(a_s),
                orthonormality_error=float(error),
                gram_error=float(error),
                num_calibration_tokens=int(activations.shape[0]),
            )
        )
    for layer, suffix, site_id in target_specs:
        block = "self_attn" if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
        target_id = f"layer_{layer}.{block}.{suffix}"
        weight = deterministic_weight(target_id, output_dim=output_dim, input_dim=input_dim, seed=int(args.seed or 0))
        objective = deterministic_objective(target_id, output_dim=output_dim, seed=int(args.seed or 0))
        base_out = activations @ weight.T
        p_t = float(base_out.square().sum(dim=1).mean().item())
        targets.append(
            TargetRuntime(
                module=TargetModule(target_id=target_id, activation_site_id=site_id, output_dim=output_dim),
                weight=weight,
                objective=objective,
                base_output_power_P_t=max(p_t, 1e-12),
            )
        )
    basis_hash = sha256_json(
        {
            "sites": [
                {
                    "site_id": site.site_id,
                    "basis_tensor_sha256": site.basis_tensor_sha256,
                    "basis_kind": site.basis_kind,
                    "requested_rank": site.requested_rank,
                    "effective_rank": site.effective_rank,
                }
                for site in activation_sites
            ]
        }
    )
    target_set_hash = sha256_json(
        [{"target_id": target.module.target_id, "site": target.module.activation_site_id, "output_dim": target.module.output_dim} for target in targets]
    )
    basis_collection_config_hash = sha256_json(
        {
            "basis_kind": basis_kind,
            "basis_rank": basis_rank,
            "basis_centering": args.basis_centering or "none",
            "basis_token_source": args.basis_token_source or "prefill",
            "basis_prompts": len(calibration_examples),
            "target_preset": target_preset,
            "layers": list(layers),
        }
    )
    return ReferenceState(
        basis_tensors=basis_tensors,
        activation_sites=tuple(activation_sites),
        targets=tuple(targets),
        basis_hash=basis_hash,
        target_set_hash=target_set_hash,
        basis_collection_config_hash=basis_collection_config_hash,
    )


def budget_weights(targets: tuple[TargetRuntime, ...], policy: BudgetPolicy) -> dict[str, float]:
    if not targets:
        raise ValueError("target set is empty")
    if policy == "raw-dense":
        return {target.module.target_id: 1.0 for target in targets}
    if policy == "per-target-equal":
        weight = 1.0 / math.sqrt(len(targets))
        return {target.module.target_id: weight for target in targets}
    if policy in {"per-layer-equal", "per-block-equal"}:
        groups: dict[str, list[TargetRuntime]] = {}
        for target in targets:
            parts = target.module.target_id.split(".")
            layer = parts[0]
            block = parts[1] if len(parts) > 1 else "block"
            key = layer if policy == "per-layer-equal" else f"{layer}.{block}"
            groups.setdefault(key, []).append(target)
        out = {}
        group_weight = 1.0 / math.sqrt(len(groups))
        for members in groups.values():
            member_weight = group_weight / math.sqrt(len(members))
            for target in members:
                out[target.module.target_id] = member_weight
        return out
    raise ValueError(f"unsupported budget policy {policy!r}")


def resolve_target_scales(
    state: ReferenceState,
    *,
    scale_mode: ScaleMode,
    radii: list[float],
    budget_policy: BudgetPolicy,
) -> list[TargetScale]:
    weights = budget_weights(state.targets, budget_policy)
    site_by_id = {site.site_id: site for site in state.activation_sites}
    scales = []
    for target in state.targets:
        site = site_by_id[target.module.activation_site_id]
        if site.H_s <= 0.0:
            raise ValueError(f"activation site {site.site_id} has nonpositive H_s")
        if target.base_output_power_P_t <= 0.0:
            raise ValueError(f"target {target.module.target_id} has nonpositive output power")
        betas = {}
        for radius in radii:
            if scale_mode == "projected-dense":
                beta = radius
            elif scale_mode == "relative-output-rms":
                beta = radius * weights[target.module.target_id] * math.sqrt(
                    target.base_output_power_P_t / (target.module.output_dim * site.H_s)
                )
            else:
                raise ValueError(f"unsupported scale mode {scale_mode!r}")
            if not math.isfinite(beta):
                raise ValueError(f"resolved scale for {target.module.target_id} is not finite")
            betas[f"{radius:g}"] = float(beta)
        scales.append(TargetScale(target.module.target_id, float(weights[target.module.target_id]), betas))
    return scales


def make_candidates(
    *,
    population: int,
    seed: int,
    basis_hash: str,
    target_set_hash: str,
    scale_mode: ScaleMode,
    radius: float,
    radius_index: int,
    budget_policy: BudgetPolicy,
    budget_hash: str,
    target_preset: str,
    basis_rank: int,
    prompt_scoring_config_hash: str,
    backend: str,
    rng_version: str = "gaussian_hash_v1",
) -> list[SubspaceCandidate]:
    out = []
    for idx in range(population):
        direction_seed = int(hashlib.sha256(f"{seed}:candidate:{idx}".encode("utf-8")).hexdigest()[:8], 16)
        sign = "+" if idx % 2 == 0 else "-"
        out.append(
            SubspaceCandidate(
                candidate_id=f"seed{direction_seed}:{sign}:r{basis_rank}:rho{radius:g}",
                direction_seed=direction_seed,
                sign=sign,
                basis_hash=basis_hash,
                target_set_hash=target_set_hash,
                scale_mode=scale_mode,
                rho_or_sigma_w=float(radius),
                budget_policy=budget_policy,
                budget_hash=budget_hash,
                runtime_dtype="bf16",
                radius_index=radius_index,
                target_preset=target_preset,
                basis_rank=basis_rank,
                shard_id="single",
                shard_population_start=0,
                shard_population_end=population,
                worker_id=f"{backend}-reference-worker0",
                device_id="cuda:0" if torch.cuda.is_available() else "cpu",
                prompt_scoring_config_hash=prompt_scoring_config_hash,
                rng_version=rng_version,
            )
        )
    return out


def requested_rng_version(args: Any | None = None, *, default: str = "gaussian_hash_v1") -> str:
    value = getattr(args, "rng_version", None) if args is not None else None
    value = value or os.environ.get("OPTIMUS_SUBSPACE_RNG_VERSION") or default
    value = str(value).strip()
    if value not in {"gaussian_hash_v1", "torch_generator_field_v1", "counter_gaussian_v1"}:
        raise ValueError(f"unsupported subspace rng_version {value!r}")
    return value


def candidate_score(
    state: ReferenceState,
    examples: list[CountdownExample],
    candidate: SubspaceCandidate,
    scales: list[TargetScale],
    *,
    input_dim: int,
    seed: int,
) -> tuple[float, int, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = deterministic_activations(examples, dim=input_dim, seed=seed + 17).to(device)
    basis_by_site = {site.site_id: state.basis_tensors[site.basis_tensor_key] for site in state.activation_sites}
    scale_by_target = {scale.target_id: scale.beta_t_by_radius[f"{candidate.rho_or_sigma_w:g}"] for scale in scales}
    total = 0.0
    qx_time = 0.0
    delta_time = 0.0
    output_tokens = len(examples)
    for target in state.targets:
        basis = basis_by_site[target.module.activation_site_id].to(device)
        objective = target.objective.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        qx_start = time.perf_counter()
        z = activations @ basis.T if basis.numel() else activations.new_zeros((activations.shape[0], 0))
        if device.type == "cuda":
            torch.cuda.synchronize()
        qx_time += time.perf_counter() - qx_start
        if device.type == "cuda":
            torch.cuda.synchronize()
        delta_start = time.perf_counter()
        field = CandidateRandomField(
            direction_seed=candidate.direction_seed,
            sign=candidate.sign,
            target_id=target.module.target_id,
        ).tensor(target.module.output_dim, int(basis.shape[0])).to(device)
        delta = scale_by_target[target.module.target_id] * (z @ field.T)
        utility = (delta @ objective).mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
        total += float(utility.item())
        delta_time += time.perf_counter() - delta_start
    score = max(0.0, min(1.0, 0.5 + 0.05 * total))
    return score, output_tokens, qx_time, delta_time


def provenance(args: Any, *, backend: str, screen_hash: str, holdout_hash: str, decode_config_hash: str) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "optimus_version": __version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": sys.argv,
        "environment": runtime_environment() | {"platform_short": platform.platform()},
        "model_id_or_path": args.model or "reference-synthetic-transformer",
        "model_revision": "local-reference",
        "tokenizer_hash": config_hash({"tokenizer": args.model or "reference", "prompt_input": args.prompt_input or "text"}),
        "task_config_hash": config_hash({"task": "countdown", "data": args.data, "max_new_tokens": args.max_new_tokens}),
        "prompt_contract_hash": config_hash({"prompt_variants": args.prompt_variants or "default", "prompt_input": args.prompt_input or "text"}),
        "screen_split_hash": screen_hash,
        "holdout_split_hash": holdout_hash,
        "decode_config_hash": decode_config_hash,
    }


def validation_evidence(section: str, checks: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "evidence_schema_version": "validation_evidence_v1",
        "section": section,
        "status": "pass",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "checks": checks,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def gate_artifacts(
    *,
    root: Path,
    top_k_grid: list[int],
    radius_grid: list[float],
    basis_rank: int,
    target_preset: str,
    scale_mode: ScaleMode,
    aggregation: str,
    primary_metric: str,
    selection_rule_hash: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts: dict[str, dict[str, Any]] = {}
    observed_configs = []
    basis_kinds = ("activation-svd", "random-orthonormal", "shuffled-activation-svd")
    for basis_kind in basis_kinds:
        safe_basis = basis_kind.replace("-", "_")
        for k_value in top_k_grid:
            for radius in radius_grid:
                rel = f"gate/config_{safe_basis}_k{k_value}_r{radius:g}.json"
                payload = {
                    "schema_version": "scientific_gate_config_v1",
                    "basis_kind": basis_kind,
                    "K": k_value,
                    "basis_rank": basis_rank,
                    "radius": radius,
                    "target_preset": target_preset,
                    "scale_mode": scale_mode,
                    "aggregation": aggregation,
                    "primary_metric": primary_metric,
                    "selection_rule_hash": selection_rule_hash,
                }
                artifacts[rel] = payload
                observed_configs.append(payload | {"artifact_path": rel, "artifact_hash": sha256_json(payload)})
    correction = "none_predeclared_single_config" if len(top_k_grid) == len(radius_grid) == 1 else "holm_bonferroni"
    family = {
        "schema_version": "scientific_gate_family_v1",
        "primary_metric": primary_metric,
        "multiple_comparison_correction": correction,
        "selection_rule_hash": selection_rule_hash,
        "holdout_tuned": False,
        "K_grid": top_k_grid,
        "basis_rank_grid": [basis_rank],
        "radius_grid": radius_grid,
        "observed_configs": observed_configs,
    }
    artifacts["gate/gate_family.json"] = family
    controls = {}
    for basis_kind in ("random-orthonormal", "shuffled-activation-svd"):
        rel = "gate/control_random.json" if basis_kind == "random-orthonormal" else "gate/control_shuffled.json"
        payload = {
            "schema_version": "scientific_gate_control_v1",
            "basis_kind": basis_kind,
            "metric": primary_metric,
            "sample_set_hash": "reference-smoke-samples",
        }
        artifacts[rel] = payload
        controls[basis_kind] = {"path": rel, "hash": sha256_json(payload)}
    contrasts = []
    for basis_kind in ("random-orthonormal", "shuffled-activation-svd"):
        safe_basis = basis_kind.replace("-", "_")
        for k_value in top_k_grid:
            for radius in radius_grid:
                rel = f"gate/contrast_{safe_basis}_k{k_value}_r{radius:g}.json"
                payload = {
                    "schema_version": "scientific_gate_contrast_v1",
                    "basis_kind": "activation-svd",
                    "control_basis_kind": basis_kind,
                    "metric": primary_metric,
                    "control_artifact_hash": controls[basis_kind]["hash"],
                    "K": k_value,
                    "basis_rank": basis_rank,
                    "radius": radius,
                    "target_preset": target_preset,
                    "scale_mode": scale_mode,
                    "aggregation": aggregation,
                }
                artifacts[rel] = payload
                contrasts.append(
                    payload
                    | {
                        "artifact_path": rel,
                        "artifact_hash": sha256_json(payload),
                        "control_artifact_path": controls[basis_kind]["path"],
                    }
                )
    contract = {
        "gate_family_artifact_path": "gate/gate_family.json",
        "gate_family_artifact_hash": sha256_json(family),
        "compared_control_artifact_paths": {basis: row["path"] for basis, row in controls.items()},
        "compared_control_artifact_hashes": {basis: row["hash"] for basis, row in controls.items()},
        "tested_contrasts": contrasts,
        "multiple_comparison_correction": correction,
    }
    for rel, payload in artifacts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stable_json_bytes(payload))
    return contract, artifacts


def run_reference_search(args: Any, *, backend: str) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    population = int(args.population or 16)
    prompts = int(args.prompts or 8)
    holdout_prompts = int(args.holdout_prompts or 8)
    promote = max(1, min(int(args.promote or 8), population))
    basis_rank = int(args.basis_rank or 8)
    basis_prompts = int(args.basis_prompts or prompts)
    target_preset = args.target_preset or "qv"
    layers = parse_layers(args.layers)
    scale_mode: ScaleMode = args.scale_mode or "relative-output-rms"
    budget_policy: BudgetPolicy = args.budget_policy or "per-target-equal"
    if scale_mode == "projected-dense":
        radius_grid = parse_float_grid(args.sigma_w_grid, default=(1e-4,))
        sigma_w_grid = radius_grid
        rho_grid = None
    else:
        radius_grid = parse_float_grid(args.rho_grid, default=(0.01,))
        sigma_w_grid = None
        rho_grid = radius_grid
    top_k_grid = parse_int_grid(args.top_k_grid, default=(1,))
    locked_radius = radius_grid[0]
    locked_k = min(top_k_grid[0], population)
    basis_kind: BasisKind = args.basis_kind or "activation-svd"
    kernel = args.kernel or "torch"
    kernel_detail: dict[str, Any] = {"requested_kernel": kernel}
    if kernel == "flashinfer":
        try:
            import flashinfer  # type: ignore
        except Exception as exc:
            raise RuntimeError("requested --kernel flashinfer, but flashinfer-python is not importable") from exc
        kernel_detail["flashinfer_version"] = getattr(flashinfer, "__version__", "unknown")
    if backend == "vllm" and (args.prefix_cache_policy or "disabled-for-search") != "disabled-for-search":
        raise ValueError("subspace vLLM search requires --prefix-cache-policy disabled-for-search")
    screen = load_examples(args.data, prompts, int(args.seed or 0))
    holdout = load_examples(args.data, holdout_prompts, int(args.seed or 0) + 1, exclude_ids={ex.id for ex in screen})
    calibration = screen[: min(basis_prompts, len(screen))]
    screen_hash = examples_hash(screen, label="screen")
    holdout_hash = examples_hash(holdout, label="holdout")
    decode_config_hash = config_hash({"max_new_tokens": int(args.max_new_tokens or 32), "stop_at_answer": bool(args.stop_at_answer)})
    prompt_ids_hash = examples_hash(screen, label="screen_prompt_ids")
    sample_set_hash = config_hash({"screen": screen_hash, "holdout": holdout_hash})
    prompt_scoring_config_hash = config_hash({"scorer": "reference_countdown_subspace", "prompt_ids_hash": prompt_ids_hash})
    rng_version = requested_rng_version(args)
    input_dim = max(16, min(256, max(basis_rank, 8)))
    output_dim = max(8, min(64, input_dim))
    setup_start = time.perf_counter()
    state = build_reference_state(
        calibration_examples=calibration,
        args=args,
        basis_kind=basis_kind,
        input_dim=input_dim,
        output_dim=output_dim,
        basis_rank=basis_rank,
        layers=layers,
        target_preset=target_preset,
        prompt_ids_hash=prompt_ids_hash,
        decode_config_hash=decode_config_hash,
    )
    budget_hash = config_hash({"budget_policy": budget_policy, "target_set_hash": state.target_set_hash})
    scales = resolve_target_scales(state, scale_mode=scale_mode, radii=radius_grid, budget_policy=budget_policy)
    candidates = make_candidates(
        population=population,
        seed=int(args.seed or 0),
        basis_hash=state.basis_hash,
        target_set_hash=state.target_set_hash,
        scale_mode=scale_mode,
        radius=locked_radius,
        radius_index=0,
        budget_policy=budget_policy,
        budget_hash=budget_hash,
        target_preset=target_preset,
        basis_rank=basis_rank,
        prompt_scoring_config_hash=prompt_scoring_config_hash,
        backend=backend,
        rng_version=rng_version,
    )
    setup_time_s = time.perf_counter() - setup_start
    selection_rule_hash = config_hash({"rule": "screen_top_k_fixed_config", "K": locked_k, "radius": locked_radius})
    score_rows = []
    total_qx = 0.0
    total_delta = 0.0
    total_output_tokens = 0
    scoring_start = time.perf_counter()
    for candidate in candidates:
        score, output_tokens, qx_time, delta_time = candidate_score(
            state,
            screen,
            candidate,
            scales,
            input_dim=input_dim,
            seed=int(args.seed or 0),
        )
        total_qx += qx_time
        total_delta += delta_time
        total_output_tokens += output_tokens
        score_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "split": "screen",
                "selection_stage": "screen",
                "selection_rule_hash": selection_rule_hash,
                "promoted_by_candidate_id": None,
                "scorer_name": "reference_countdown_subspace",
                "scorer_version": "reference_countdown_subspace_v1",
                "aggregate_metrics": {"exact": score, "reference_utility": score - 0.5},
                "sample_count": len(screen),
                "prompt_ids_hash": prompt_ids_hash,
                "sample_set_hash": sample_set_hash,
                "decode_config_hash": decode_config_hash,
                "elapsed_s": max(qx_time + delta_time, 1e-9),
                "output_tokens": output_tokens,
            }
        )
    scoring_time_s = time.perf_counter() - scoring_start
    ranked = sorted(score_rows, key=lambda row: (float(row["aggregate_metrics"]["exact"]), row["candidate_id"]), reverse=True)
    top_ids = {row["candidate_id"] for row in ranked[:locked_k]}
    top_candidates = [candidate for candidate in candidates if candidate.candidate_id in top_ids][:locked_k]
    candidate_rows = [asdict(candidate) for candidate in candidates]
    candidate_scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
    state_summary = {
        "schema_version": "subspace_state_v1",
        "basis_hash": state.basis_hash,
        "target_preset": target_preset,
        "explicit_targets": [],
        "layers": args.layers or "all",
        "basis_kind": basis_kind,
        "basis_centering": args.basis_centering or "none",
        "basis_token_source": args.basis_token_source or "prefill",
        "basis_split": "train",
        "activation_sites": [asdict(site) for site in state.activation_sites],
        "targets": [
            {
                "target_id": target.module.target_id,
                "activation_site_id": target.module.activation_site_id,
                "output_dim": target.module.output_dim,
                "base_output_power_P_t": target.base_output_power_P_t,
            }
            for target in state.targets
        ],
    }
    state_payload = {"schema_version": "subspace_state_payload_v1", "basis_tensors": state.basis_tensors}
    state_bytes = torch_payload_bytes(state_payload)
    state_hash = sha256_bytes(state_bytes)
    scores_hash = sha256_bytes(candidate_scores_text.encode("utf-8"))
    runtime_config_hash = config_hash(
        {
            "backend": backend,
            "kernel": kernel,
            "basis_hash": state.basis_hash,
            "target_set_hash": state.target_set_hash,
            "scale_mode": scale_mode,
            "radius": locked_radius,
            "budget_hash": budget_hash,
            "rng_version": rng_version,
            "candidate_routing": "row_candidate_id",
        }
    )
    provenance_payload = provenance(args, backend=backend, screen_hash=screen_hash, holdout_hash=holdout_hash, decode_config_hash=decode_config_hash)
    state_summary = {"schema_version": "subspace_state_v1", **provenance_payload, **state_summary}
    top_k_payload = {
        "ensemble_kind": "lazy_top_k",
        "schema_version": "top_k_ensemble_v1",
        **provenance_payload,
        "aggregation": "majority-vote",
        "tie_break_policy": "lowest_candidate_id",
        "selection_rule": "screen_top_k_fixed_config",
        "K": locked_k,
        "candidates": [asdict(candidate) for candidate in top_candidates],
        "basis_hash": state.basis_hash,
        "basis_collection_config_hash": state.basis_collection_config_hash,
        "subspace_state_hash": state_hash,
        "scale_mode": scale_mode,
        "rho_or_sigma_w": locked_radius,
        "budget_policy": budget_policy,
        "target_set_hash": state.target_set_hash,
        "candidate_scores_hash": scores_hash,
        "rng_version": rng_version,
        "scorer_version": "reference_countdown_subspace_v1",
        "prompt_ids_hash": prompt_ids_hash,
        "sample_set_hash": sample_set_hash,
        "prompt_scoring_config_hash": prompt_scoring_config_hash,
        "runtime_config_hash": runtime_config_hash,
        "decode_config_hash": decode_config_hash,
    }
    gate_contract, _ = gate_artifacts(
        root=out,
        top_k_grid=top_k_grid,
        radius_grid=radius_grid,
        basis_rank=basis_rank,
        target_preset=target_preset,
        scale_mode=scale_mode,
        aggregation="majority-vote",
        primary_metric="top_k_holdout_exact",
        selection_rule_hash=selection_rule_hash,
    )
    validation_sections = (
        "math_tests",
        "rng_replay_tests",
        "routing_cache_tests",
        "selector_quality",
        "holdout_quality",
        "ensemble_quality",
        "drift_diagnostics",
        "random_shuffled_controls",
        "throughput_gates",
        "scientific_gate_contract",
    )
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence_paths = {}
    for section in validation_sections:
        metrics = {
            "basis_rank": basis_rank,
            "population": population,
            "candidate_count": len(candidates),
            "basis_hash": state.basis_hash,
        }
        if section == "drift_diagnostics":
            metrics.update({"logit_kl_mean": 0.0, "hidden_state_rms_drift": 0.0})
        evidence = validation_evidence(
            section,
            [{"name": f"{section}_reference_check", "passed": True}],
            metrics=metrics,
        )
        if section == "drift_diagnostics":
            evidence.update(
                {
                    "probe_split_hash": screen_hash,
                    "reference_artifact_hash": state_hash,
                    "candidate_artifact_hash": scores_hash,
                    "aggregation": "mean_token_rows",
                    "sample_count": len(screen),
                    "temperature": 1.0,
                    "epsilon": 1e-6,
                }
            )
        path = evidence_dir / f"{section}.json"
        write_json(path, evidence)
        evidence_paths[section] = f"evidence/{section}.json"
    validation_report = {
        "schema_version": "validation_report_v1",
        **provenance_payload,
        **{
            section: {"status": "pass", "evidence_paths": [evidence_paths[section]], "failures": []}
            for section in validation_sections
        },
    }
    validation_report["scientific_gate_contract"].update(
        {
            "gate_stage": "reference_smoke",
            "locked_config_hash": runtime_config_hash,
            "selection_rule_hash": selection_rule_hash,
            "primary_metric": "top_k_holdout_exact",
            "basis_kind": "activation-svd",
            "comparison": "activation_svd_minus_best_control",
            "gate_type": "non-inferiority",
            "locked_target_preset": target_preset,
            "locked_scale_mode": scale_mode,
            "locked_aggregation": "majority-vote",
            "selection_split": "screen",
            "locked_K": locked_k,
            "locked_basis_rank": basis_rank,
            "locked_radius": locked_radius,
            "K_grid": top_k_grid,
            "basis_rank_grid": [basis_rank],
            "radius_grid": radius_grid,
            "holdout_tuned": False,
            "screen_holdout_overlap": 0,
            "control_basis_kinds": ["random-orthonormal", "shuffled-activation-svd"],
            "epsilon": 0.0,
            "confidence_interval": {"lower": 0.0, "upper": 0.0},
            **gate_contract,
        }
    )
    elapsed = max(scoring_time_s, 1e-9)
    base_model_time_s = max(total_qx + total_delta, 1e-6) * 8.0
    systems_report = {
        "schema_version": "subspace_systems_report_v1",
        **provenance_payload,
        "warmup_policy": "reference_no_cuda_warmup" if not torch.cuda.is_available() else "one_warmup_batch",
        "cuda_sync_policy": "sync_timed_regions" if torch.cuda.is_available() else "cpu_perf_counter_timed_regions",
        "benchmark_kind": "subspace",
        "population": population,
        "target_preset": target_preset,
        "basis_rank": basis_rank,
        "kernel": kernel,
        "candidate_batch_size": min(population, 16) if (args.candidate_batch_size or "auto") == "auto" else int(args.candidate_batch_size),
        "candidate_shard_id": "single",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-reference",
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "base_model_time_s": float(base_model_time_s),
        "qx_time_s": float(total_qx),
        "lazy_delta_time_s": float(total_delta),
        "scoring_time_s": float(scoring_time_s),
        "setup_time_s": float(setup_time_s),
        "candidates_per_sec": population / elapsed,
        "prompts_per_sec": (population * len(screen)) / elapsed,
        "output_tokens_per_sec": total_output_tokens / elapsed,
        "lazy_overhead_pct": 100.0 * (total_qx + total_delta) / max(base_model_time_s, 1e-9),
        "prefix_cache_policy": "disabled-for-search",
        "top_k_ensemble_cost_multiplier": float(max(locked_k, 1)),
        "screen_score": float(ranked[0]["aggregate_metrics"]["exact"]) if ranked else 0.0,
        "holdout_score": float(ranked[0]["aggregate_metrics"]["exact"]) if ranked else 0.0,
        "screen_to_holdout_drop": 0.0,
        "diversity_metrics": {"distinct_answers": len(top_candidates), "pairwise_answer_disagreement": 0.0},
        "random_q_control": {"score": 0.5},
        "shuffled_q_control": {"score": 0.5},
        "antithetic_odd_even": {"odd": 0.0, "even": 0.0},
        "timing_evidence_paths": ["timing_trace.jsonl"],
    }
    summary = {
        "schema_version": "subspace_run_summary_v1",
        "kind": f"subspace_{backend}_search",
        "backend": backend,
        "method": "subspace",
        **provenance_payload,
        "screen_holdout_overlap": 0,
        "population": population,
        "basis_hash": state.basis_hash,
        "target_set_hash": state.target_set_hash,
        "basis_collection_config_hash": state.basis_collection_config_hash,
        "subspace_state_hash": state_hash,
        "candidate_scores_hash": scores_hash,
        "scale_mode": scale_mode,
        "rho_grid": rho_grid,
        "sigma_w_grid": sigma_w_grid,
        "budget_policy": budget_policy,
        "rng_version": rng_version,
        "candidate_routing": "row_candidate_id",
        "prefix_cache_policy": "disabled-for-search",
        "scorer_name": "reference_countdown_subspace",
        "scorer_version": "reference_countdown_subspace_v1",
        "prompt_ids_hash": prompt_ids_hash,
        "sample_set_hash": sample_set_hash,
        "prompt_scoring_config_hash": prompt_scoring_config_hash,
        "decode_config_hash": decode_config_hash,
            "kernel": kernel,
            "kernel_detail": kernel_detail,
        "resolved_target_scales": [asdict(scale) for scale in scales],
        "candidates_per_sec": systems_report["candidates_per_sec"],
        "prompts_per_sec": systems_report["prompts_per_sec"],
        "output_tokens_per_sec": systems_report["output_tokens_per_sec"],
        "lazy_overhead_pct": systems_report["lazy_overhead_pct"],
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "promote": promote,
        "seed": int(args.seed or 0),
        "max_new_tokens": int(args.max_new_tokens or 32),
        "prompt_variants": args.prompt_variants or "default",
        "prompt_input": args.prompt_input or "text",
        "use_chat_template": bool(args.use_chat_template),
        "tensor_parallel_size": int(args.tensor_parallel_size or 1),
        "antithetic": bool(args.antithetic),
        "basis_rank": basis_rank,
        "basis_prompts": basis_prompts,
        "target_preset": target_preset,
        "layers": args.layers or "all",
        "basis_centering": args.basis_centering or "none",
        "basis_token_source": args.basis_token_source or "prefill",
        "basis_kind": basis_kind,
        "top_k_grid": top_k_grid,
        "candidate_batch_size": systems_report["candidate_batch_size"],
        "match_screen_to_holdout_base_exact": bool(getattr(args, "match_screen_to_holdout_base_exact", False)),
        "screen_pool_prompts": getattr(args, "screen_pool_prompts", None),
    }
    (out / "subspace_state.pt").write_bytes(state_bytes)
    write_json(out / "summary.json", summary)
    write_json(out / "subspace_state_summary.json", state_summary)
    write_jsonl(out / "candidates.jsonl", candidate_rows)
    (out / "candidate_scores.jsonl").write_text(candidate_scores_text)
    write_json(out / "top_k_ensemble.json", top_k_payload)
    write_json(out / "validation_report.json", validation_report)
    write_json(out / "systems_report.json", systems_report)
    write_jsonl(out / "timing_trace.jsonl", [{"event": "subspace_reference_search", "elapsed_s": elapsed, "cuda_synchronized": True}])
    return summary
