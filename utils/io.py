"""Stable CSV writing helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "schema_version",
    "run_id",
    "timestamp",
    "gpu_name",
    "cuda_version",
    "torch_version",
    "triton_version",
    "dtype",
    "experiment",
    "projection",
    "mode",
    "provider",
    "parallel_type",
    "parallel_size",
    "num_experts",
    "topk",
    "T",
    "M",
    "K",
    "N",
    "num_gemms",
    "active_gemms",
    "total_flops",
    "latency_ms",
    "latency_p20_ms",
    "latency_p80_ms",
    "tflops",
    "normalized_efficiency",
    "peak_efficiency",
    "arithmetic_intensity",
    "correct",
    "max_abs_error",
    "max_rel_error",
    "cache_mode",
    "workspace_count",
    "workspace_bytes",
    "tokens_per_expert",
    "weight_shard_factor",
    "flops_equal",
    "autotune_enabled",
    "autotune_status",
    "config_source",
    "block_m",
    "block_n",
    "block_k",
    "group_size_m",
    "num_warps",
    "num_stages",
    "cta_multiplier",
    "scheduler",
    "autotune_candidates_tested",
    "autotune_candidates_failed",
    "autotune_selection_ms",
    "best_measured_tflops",
    "best_measured_ratio",
    "near_best_threshold",
    "near_best",
    "output_tiles_per_gemm",
    "total_output_tiles_per_rank",
    "dot_tiles",
    "tiles_per_sm",
    "estimated_waves",
    "tile_efficiency",
    "output_tile_efficiency",
    "launches_per_iteration",
    "warmup",
    "repeat",
    "requested_warmup",
    "requested_repeat",
    "estimated_memory_bytes",
    "memory_budget_bytes",
    "model_config",
    "model_type",
    "requested_parameters",
    "effective_parameters",
    "status",
    "skip_reason",
    "error",
]


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
