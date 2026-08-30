"""Pure mathematical helpers used by benchmarks and tests."""

from __future__ import annotations

import math
from collections.abc import Iterable


def gemm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def arithmetic_intensity(m: int, k: int, n: int, element_size: int) -> float:
    """Ideal one-read/one-write traffic model, not measured HBM traffic."""
    byte_count = element_size * (m * k + k * n + m * n)
    return gemm_flops(m, k, n) / byte_count if byte_count else 0.0


def tile_metrics(
    shapes: Iterable[tuple[int, int, int]], block_m: int, block_n: int, block_k: int, sm_count: int
) -> dict[str, float | int]:
    shapes = list(shapes)
    output_tiles = sum(math.ceil(m / block_m) * math.ceil(n / block_n) for m, _, n in shapes)
    dot_tiles = sum(math.ceil(m / block_m) * math.ceil(n / block_n) * math.ceil(k / block_k) for m, k, n in shapes)
    useful_flops = sum(gemm_flops(m, k, n) for m, k, n in shapes)
    padded_flops = dot_tiles * 2 * block_m * block_n * block_k
    useful_outputs = sum(m * n for m, _, n in shapes)
    padded_outputs = output_tiles * block_m * block_n
    return {
        "output_tiles_per_gemm": output_tiles / len(shapes) if shapes else 0.0,
        "total_output_tiles_per_rank": output_tiles,
        "dot_tiles": dot_tiles,
        "tiles_per_sm": output_tiles / sm_count if sm_count else 0.0,
        "estimated_waves": math.ceil(output_tiles / sm_count) if sm_count and output_tiles else 0,
        "tile_efficiency": useful_flops / padded_flops if padded_flops else 0.0,
        "output_tile_efficiency": useful_outputs / padded_outputs if padded_outputs else 0.0,
    }


def moe_shapes(
    projection: str,
    parallel_type: str,
    m: int,
    hidden: int,
    ffn: int,
    experts: int,
    parallel_size: int,
) -> list[tuple[int, int, int]]:
    """Build one rank's expert GEMMs after an assumed uniform routing step.

    ``m`` is the number of top-k-expanded assignment rows per expert. Therefore
    a TP rank represents ``experts * m`` input rows and an EP rank represents
    ``experts * m / parallel_size`` rows. Routing and permutation are not part
    of this shape-only construction.
    """
    if experts % parallel_size:
        raise ValueError("num_experts must be divisible by parallel_size")
    if ffn % parallel_size:
        raise ValueError("ffn_size must be divisible by parallel_size")
    count = experts // parallel_size if parallel_type == "EP" else experts
    if projection == "W1":
        shape = (m, hidden, ffn if parallel_type == "EP" else ffn // parallel_size)
    elif projection == "W2":
        shape = (m, ffn if parallel_type == "EP" else ffn // parallel_size, hidden)
    else:
        raise ValueError(f"unknown projection {projection}")
    return [shape] * count


def verify_ep_tp_flops(projection: str, m: int, hidden: int, ffn: int, experts: int, p: int) -> int:
    ep = sum(gemm_flops(*shape) for shape in moe_shapes(projection, "EP", m, hidden, ffn, experts, p))
    tp = sum(gemm_flops(*shape) for shape in moe_shapes(projection, "TP", m, hidden, ffn, experts, p))
    if ep != tp:
        raise ValueError(f"per-rank FLOPs differ: EP={ep}, TP={tp}")
    return ep
