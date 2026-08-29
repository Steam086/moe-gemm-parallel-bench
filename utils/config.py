"""Model configuration loading and benchmark defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelDefaults:
    model_path: str
    model_type: str
    hidden_size: int
    ffn_size: int
    num_experts: int
    topk: int
    dtype: str
    config_ep_size: int
    quantization: dict[str, Any] | None


def _first(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def load_model_defaults(path: str | Path) -> ModelDefaults:
    """Load common MoE config schemas with explicit, actionable errors.

    Multimodal configs place language-model fields under ``text_config``.
    A dedicated MoE intermediate width is preferred; ``intermediate_size`` is
    used only for schemas where it is the published expert width.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    text = raw.get("text_config")
    scope = text if isinstance(text, dict) else raw
    values = {
        "hidden_size": _first(scope, ("hidden_size",)),
        "ffn_size": _first(scope, ("moe_intermediate_size", "intermediate_size")),
        "num_experts": _first(scope, ("n_routed_experts", "num_local_experts", "num_experts")),
        "topk": _first(scope, ("num_experts_per_tok", "num_experts_per_token", "router_top_k")),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"unsupported MoE config schema in {path}: missing " + ", ".join(missing))
    declared_dtype = _first(scope, ("torch_dtype", "dtype")) or _first(raw, ("torch_dtype", "dtype")) or "bfloat16"
    dtype = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(str(declared_dtype), "bf16")
    quantization = scope.get("quantization_config") or raw.get("quantization_config")
    return ModelDefaults(
        model_path=str(path),
        model_type=str(raw.get("model_type", scope.get("model_type", path.stem))),
        hidden_size=int(values["hidden_size"]),
        ffn_size=int(values["ffn_size"]),
        num_experts=int(values["num_experts"]),
        topk=int(values["topk"]),
        dtype=dtype,
        config_ep_size=int(scope.get("ep_size", raw.get("ep_size", 1))),
        quantization=quantization,
    )
