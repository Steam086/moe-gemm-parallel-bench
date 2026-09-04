"""Persistent grouped GEMM over distinct expert A/B/C pointers.

Equal, contiguous MoE problems use a constexpr-shape fast path. Arbitrary
shape/stride groups use a device-side scheduled fallback with a per-problem K
loop, so a short-K problem never executes dot operations up to the group's
maximum K.  Gate+Up uses the same kernel with a packed W13 output dimension:
the first N/2 columns are Gate and the second N/2 columns are Up, matching the
layout used by vLLM's Triton FusedMoE path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    torch = None
    triton = None
    tl = None

from .matmul import (
    _MATMUL_CONFIG_POOL,
    KernelConfig,
    last_matmul_config,
    launch_matmul,
    matmul_candidate_configs,
    select_config,
)

# Extra portable tiles for skinny-M MoE groups. They deliberately remain a
# small curated extension of the standalone GEMM pool so Triton compilation
# and autotuning stay bounded.
_GROUPED_EXTRA_CONFIG_POOL: tuple[KernelConfig, ...] = (
    KernelConfig(16, 64, 64, 4, 4, 3),
    KernelConfig(16, 128, 128, 4, 4, 4),
    KernelConfig(16, 256, 32, 4, 8, 3),
    KernelConfig(16, 256, 64, 4, 8, 4),
    KernelConfig(32, 64, 128, 4, 4, 4),
    KernelConfig(32, 128, 128, 4, 8, 4),
    KernelConfig(32, 256, 64, 4, 8, 4),
    KernelConfig(64, 32, 64, 4, 4, 3),
    KernelConfig(64, 256, 128, 4, 8, 4),
    KernelConfig(128, 128, 128, 8, 8, 4),
)
_GROUPED_BASE_CONFIG_POOL = tuple(dict.fromkeys((*_MATMUL_CONFIG_POOL, *_GROUPED_EXTRA_CONFIG_POOL)))


@dataclass
class GroupedWorkspace:
    a: list
    b: list
    c: list
    a_ptrs: object
    b_ptrs: object
    c_ptrs: object
    dims: object
    strides: object
    shapes: tuple[tuple[int, int, int], ...]
    config: KernelConfig
    shape_signature: int
    device_signature: int
    sm_count: int
    storage_bytes: int
    homogeneous: bool
    operation: str = "gemm"
    scheduler: str = "not_launched"

    @property
    def problem_count(self) -> int:
        return len(self.a)

    @property
    def max_m(self) -> int:
        return max(m for m, _, _ in self.shapes)

    @property
    def max_n(self) -> int:
        return max(n for _, _, n in self.shapes)

    @property
    def max_k(self) -> int:
        return max(k for _, k, _ in self.shapes)

    def tile_count_for(self, config: KernelConfig) -> int:
        return sum(math.ceil(m / config.block_m) * math.ceil(n / config.block_n) for m, _, n in self.shapes)

    @property
    def tile_count(self) -> int:
        return self.tile_count_for(self.config)

    def problem_tensors(self):
        return zip(self.a, self.b, self.c)


def grouped_candidate_configs(
    max_m: int,
    max_n: int,
    max_k: int,
    base_limit: int = 8,
    *,
    problem_count: int = 1,
    sm_count: int = 0,
) -> tuple[KernelConfig, ...]:
    """Workload-aware tile and persistent-grid candidates.

    The grouped MoE path needs a broader small-M search than a standalone
    GEMM: a narrower N tile can expose enough independent expert tiles to fill
    the device even when padding efficiency alone would rank it lower. The
    final choice is measured by Triton's autotuner (hot mode) or by the
    rotating-workspace tuner (cold mode); this function only keeps that search
    bounded and avoids CTA-count variants that launch the same grid.
    """
    if min(max_m, max_n, max_k, problem_count) <= 0:
        raise ValueError("grouped candidate dimensions and problem_count must be positive")
    if sm_count < 0:
        raise ValueError("sm_count must be non-negative")

    # A one-problem group is dispatched to the standard matmul kernel, so CTA
    # persistence is not a tuning dimension for that case.
    if problem_count == 1:
        return tuple(
            KernelConfig(
                cfg.block_m,
                cfg.block_n,
                cfg.block_k,
                cfg.group_size_m,
                cfg.num_warps,
                cfg.num_stages,
                1,
            )
            for cfg in matmul_candidate_configs(max_m, max_n, max_k, limit=base_limit)
        )

    # These additions target the skinny-M MoE shapes that the standalone GEMM
    # family intentionally prunes more aggressively. In particular, N=64
    # tiles can provide useful parallelism for TP shards and wide W2 outputs.
    if max_m <= 32:
        block_ms = {16, 32, 64}
    elif max_m <= 64:
        block_ms = {32, 64, 128}
    elif max_m <= 128:
        block_ms = {64, 128}
    else:
        block_ms = {64, 128, 256}
    if max_n <= 32:
        block_ns = {32, 64}
    elif max_n <= 64:
        block_ns = {32, 64, 128}
    elif max_n <= 128:
        block_ns = {64, 128, 256}
    else:
        # Keep N=64 for small-M grouped workloads: the extra output tiles can
        # matter more than padding efficiency when expert count is modest.
        block_ns = {64, 128, 256} if max_m <= 64 else {128, 256}

    base = [
        cfg
        for cfg in _GROUPED_BASE_CONFIG_POOL
        if cfg.block_m in block_ms
        and cfg.block_n in block_ns
        and (cfg.block_k <= max_k or cfg.block_k == 32)
    ]
    fallback = select_config(max_m, max_n, max_k)
    if fallback not in base:
        base.append(fallback)

    def rank(cfg: KernelConfig) -> tuple[float, float, int, int, int]:
        m_tiles = math.ceil(max_m / cfg.block_m)
        n_tiles = math.ceil(max_n / cfg.block_n)
        k_tiles = math.ceil(max_k / cfg.block_k)
        padded = m_tiles * cfg.block_m * n_tiles * cfg.block_n * k_tiles * cfg.block_k
        useful = max_m * max_n * max_k
        compute_efficiency = useful / padded
        total_tiles = problem_count * m_tiles * n_tiles
        # Prefer enough independent work to cover the device, but let measured
        # autotuning decide whether the extra/smaller tiles are actually faster.
        parallelism = min(1.0, total_tiles / sm_count) if sm_count else 1.0
        score = compute_efficiency * (0.85 + 0.15 * parallelism)
        return (-score, -parallelism, cfg.block_m * cfg.block_n, cfg.block_k, cfg.num_warps)

    ordered = sorted(dict.fromkeys(base), key=rank)

    # Preserve diversity across the dimensions that materially change tensor
    # core utilization and occupancy, then fill the remaining bounded budget by
    # the analytical rank above. Actual hardware timing still chooses the
    # winner.
    selected_base: list[KernelConfig] = []

    def add_first(predicate) -> None:
        candidate = next((cfg for cfg in ordered if predicate(cfg)), None)
        if candidate is not None and candidate not in selected_base:
            selected_base.append(candidate)

    for block_m in sorted(block_ms):
        add_first(lambda cfg, value=block_m: cfg.block_m == value)
    for block_n in sorted(block_ns):
        add_first(lambda cfg, value=block_n: cfg.block_n == value)
    for block_k in (32, 64, 128):
        if block_k <= max_k:
            add_first(lambda cfg, value=block_k: cfg.block_k == value)
    for warps in (2, 4, 8):
        add_first(lambda cfg, value=warps: cfg.num_warps == value)
    for cfg in ordered:
        if cfg not in selected_base:
            selected_base.append(cfg)
        if len(selected_base) >= max(1, base_limit):
            break
    selected_base = selected_base[: max(1, base_limit)]

    candidates: list[KernelConfig] = []
    for cfg in selected_base:
        total_tiles = problem_count * math.ceil(max_m / cfg.block_m) * math.ceil(max_n / cfg.block_n)
        multipliers = [1]
        if not sm_count or total_tiles > sm_count:
            multipliers.append(2)
        if cfg.block_m <= 32 and (not sm_count or total_tiles > 2 * sm_count):
            multipliers.append(4)
        candidates.extend(
            KernelConfig(
                cfg.block_m,
                cfg.block_n,
                cfg.block_k,
                cfg.group_size_m,
                cfg.num_warps,
                cfg.num_stages,
                cta_multiplier,
            )
            for cta_multiplier in multipliers
        )
    return tuple(dict.fromkeys(candidates))


def _shape_signature(shapes: Sequence[tuple[int, int, int]]) -> int:
    """Stable bounded signature used only as an autotune-cache discriminator."""
    value = 2166136261
    for shape in shapes:
        for dimension in shape:
            value = ((value ^ int(dimension)) * 16777619) & 0x7FFFFFFF
    return value


def _runtime_ready() -> bool:
    if triton is None or torch is None or not torch.cuda.is_available():
        return False
    try:
        triton.runtime.driver.active.get_current_target()
        return True
    except Exception:
        return False


def _config_key(config: KernelConfig) -> tuple[int, ...]:
    return (
        config.block_m,
        config.block_n,
        config.block_k,
        config.group_size_m,
        config.num_warps,
        config.num_stages,
        config.cta_multiplier,
    )


_homogeneous_grouped_kernel_impl = None
_generic_grouped_kernel_impl = None
_homogeneous_grouped_kernel = None
_generic_grouped_kernel = None

if _runtime_ready():
    # Build the decorator's union once; shape-family pruning below selects only
    # the relevant subset for each launch.
    _grouped_pool = tuple(
        KernelConfig(
            cfg.block_m,
            cfg.block_n,
            cfg.block_k,
            cfg.group_size_m,
            cfg.num_warps,
            cfg.num_stages,
            multiplier,
        )
        for cfg in _GROUPED_BASE_CONFIG_POOL
        for multiplier in ((1, 2, 4) if cfg.block_m <= 32 else (1, 2))
    )
    _grouped_triton_configs = [
        triton.Config(
            {
                "BLOCK_M": cfg.block_m,
                "BLOCK_N": cfg.block_n,
                "BLOCK_K": cfg.block_k,
                "GROUP_M": cfg.group_size_m,
                "CTA_MULTIPLIER": cfg.cta_multiplier,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        for cfg in _grouped_pool
    ]

    def _prune_grouped_configs(configs, named_args, **kwargs):
        arguments = {**named_args, **kwargs}
        wanted = {
            _config_key(cfg)
            for cfg in grouped_candidate_configs(
                int(arguments["MAX_M"]),
                int(arguments["MAX_N"]),
                int(arguments["MAX_K"]),
                problem_count=int(arguments.get("PROBLEM_COUNT", arguments.get("problem_count", 1))),
                sm_count=int(arguments["SM_COUNT"]),
            )
        }
        return [
            cfg
            for cfg in configs
            if (
                int(cfg.kwargs["BLOCK_M"]),
                int(cfg.kwargs["BLOCK_N"]),
                int(cfg.kwargs["BLOCK_K"]),
                int(cfg.kwargs["GROUP_M"]),
                int(cfg.num_warps),
                int(cfg.num_stages),
                int(cfg.kwargs["CTA_MULTIPLIER"]),
            )
            in wanted
        ]

    @triton.jit
    def _homogeneous_grouped_kernel_impl(
        a_ptrs,
        b_ptrs,
        c_ptrs,
        a_anchor,
        b_anchor,
        c_anchor,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MAX_M: tl.constexpr,
        MAX_N: tl.constexpr,
        MAX_K: tl.constexpr,
        PROBLEM_COUNT: tl.constexpr,
        SHAPE_SIGNATURE: tl.constexpr,
        DEVICE_SIGNATURE: tl.constexpr,
        SM_COUNT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        CTA_MULTIPLIER: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        grid_size = tl.num_programs(0) + CTA_MULTIPLIER * 0 + SM_COUNT * 0
        num_m_tiles = tl.cdiv(M, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        tiles_per_problem = num_m_tiles * num_n_tiles
        total_tiles = PROBLEM_COUNT * tiles_per_problem
        width = GROUP_M * num_n_tiles
        while tile_id < total_tiles:
            problem = tile_id // tiles_per_problem
            local_tile = tile_id - problem * tiles_per_problem
            group_id = local_tile // width
            first_m = group_id * GROUP_M
            group_m = tl.minimum(num_m_tiles - first_m, GROUP_M)
            pid_m = first_m + (local_tile % group_m)
            pid_n = (local_tile % width) // group_m

            a_addr = tl.load(a_ptrs + problem)
            b_addr = tl.load(b_ptrs + problem)
            c_addr = tl.load(c_ptrs + problem)
            a_ptr = tl.cast(a_addr, tl.pointer_type(a_anchor.dtype.element_ty))
            b_ptr = tl.cast(b_addr, tl.pointer_type(b_anchor.dtype.element_ty))
            c_ptr = tl.cast(c_addr, tl.pointer_type(c_anchor.dtype.element_ty))

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            # Keep true consecutive indices. Replacing edge lanes with zero
            # invalidates max_contiguous hints and can miscompile vector loads.
            # constexpr divisibility removes M/N predicates for aligned shapes.
            a_tile = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            b_tile = b_ptr + offs_k[:, None] * N + offs_n[None, :]
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                a = tl.load(
                    a_tile,
                    mask=((M % BLOCK_M == 0) | (offs_m[:, None] < M)) & (k_start + offs_k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    b_tile,
                    mask=(k_start + offs_k[:, None] < K) & ((N % BLOCK_N == 0) | (offs_n[None, :] < N)),
                    other=0.0,
                )
                accumulator += tl.dot(a, b, input_precision=INPUT_PRECISION)
                a_tile += BLOCK_K
                b_tile += BLOCK_K * N
            tl.store(
                c_ptr + offs_m[:, None] * N + offs_n[None, :],
                accumulator,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
            tile_id += grid_size

    @triton.jit
    def _generic_grouped_kernel_impl(
        a_ptrs,
        b_ptrs,
        c_ptrs,
        dims,
        strides,
        a_anchor,
        b_anchor,
        c_anchor,
        problem_count,
        MAX_M: tl.constexpr,
        MAX_N: tl.constexpr,
        MAX_K: tl.constexpr,
        SHAPE_SIGNATURE: tl.constexpr,
        DEVICE_SIGNATURE: tl.constexpr,
        SM_COUNT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        CTA_MULTIPLIER: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        grid_size = tl.num_programs(0) + CTA_MULTIPLIER * 0 + SM_COUNT * 0
        last_problem_end = 0
        for problem in range(problem_count):
            m = tl.load(dims + problem * 3)
            n = tl.load(dims + problem * 3 + 1)
            k = tl.load(dims + problem * 3 + 2)
            num_m_tiles = tl.cdiv(m, BLOCK_M)
            num_n_tiles = tl.cdiv(n, BLOCK_N)
            problem_tiles = num_m_tiles * num_n_tiles
            problem_end = last_problem_end + problem_tiles
            width = GROUP_M * num_n_tiles

            while (tile_id >= last_problem_end) & (tile_id < problem_end):
                local_tile = tile_id - last_problem_end
                group_id = local_tile // width
                first_m = group_id * GROUP_M
                group_m = tl.minimum(num_m_tiles - first_m, GROUP_M)
                pid_m = first_m + (local_tile % group_m)
                pid_n = (local_tile % width) // group_m

                stride_am = tl.load(strides + problem * 6)
                stride_ak = tl.load(strides + problem * 6 + 1)
                stride_bk = tl.load(strides + problem * 6 + 2)
                stride_bn = tl.load(strides + problem * 6 + 3)
                stride_cm = tl.load(strides + problem * 6 + 4)
                stride_cn = tl.load(strides + problem * 6 + 5)
                a_addr = tl.load(a_ptrs + problem)
                b_addr = tl.load(b_ptrs + problem)
                c_addr = tl.load(c_ptrs + problem)
                a_ptr = tl.cast(a_addr, tl.pointer_type(a_anchor.dtype.element_ty))
                b_ptr = tl.cast(b_addr, tl.pointer_type(b_anchor.dtype.element_ty))
                c_ptr = tl.cast(c_addr, tl.pointer_type(c_anchor.dtype.element_ty))

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # This loop bound is the current problem's K. It is intentionally
                # dynamic: heterogeneous short-K problems do no extra tl.dot work.
                for kk in range(tl.cdiv(k, BLOCK_K)):
                    k_start = kk * BLOCK_K
                    a = tl.load(
                        a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak,
                        mask=(offs_m[:, None] < m) & (k_start + offs_k[None, :] < k),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(k_start + offs_k[:, None] < k) & (offs_n[None, :] < n),
                        other=0.0,
                    )
                    accumulator += tl.dot(a, b, input_precision=INPUT_PRECISION)
                tl.store(
                    c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                    accumulator,
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                )
                tile_id += grid_size
            last_problem_end = problem_end

    _common_key = [
        "MAX_M",
        "MAX_N",
        "MAX_K",
        "SHAPE_SIGNATURE",
        "DEVICE_SIGNATURE",
        "SM_COUNT",
        "INPUT_PRECISION",
    ]
    _homogeneous_grouped_kernel = triton.autotune(
        configs=_grouped_triton_configs,
        key=["M", "N", "K", "PROBLEM_COUNT", *_common_key],
        prune_configs_by={"early_config_prune": _prune_grouped_configs},
    )(_homogeneous_grouped_kernel_impl)
    _generic_grouped_kernel = triton.autotune(
        configs=_grouped_triton_configs,
        key=["problem_count", *_common_key],
        prune_configs_by={"early_config_prune": _prune_grouped_configs},
    )(_generic_grouped_kernel_impl)


def build_grouped_workspace(
    shapes: Sequence[tuple[int, int, int]],
    dtype,
    seed: int = 0,
    config: KernelConfig | None = None,
) -> GroupedWorkspace:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    if not shapes:
        raise ValueError("grouped GEMM requires at least one problem")
    normalized = tuple(tuple(int(value) for value in shape) for shape in shapes)
    if any(len(shape) != 3 or any(value <= 0 for value in shape) for shape in normalized):
        raise ValueError("every grouped shape must be a positive (M, K, N) triple")
    first_m, first_k, first_n = normalized[0]
    config = config or select_config(first_m, first_n, first_k)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    arrays_a, arrays_b, arrays_c = [], [], []
    for m, k, n in normalized:
        arrays_a.append(torch.empty((m, k), device="cuda", dtype=dtype).normal_(generator=generator).mul_(0.1))
        arrays_b.append(torch.empty((k, n), device="cuda", dtype=dtype).normal_(generator=generator).mul_(0.1))
        arrays_c.append(torch.empty((m, n), device="cuda", dtype=dtype))
    device = arrays_a[0].device

    def pointer_tensor(arrays):
        return torch.tensor([item.data_ptr() for item in arrays], device=device, dtype=torch.int64)

    a_ptrs = pointer_tensor(arrays_a)
    b_ptrs = pointer_tensor(arrays_b)
    c_ptrs = pointer_tensor(arrays_c)
    dims = torch.tensor([[m, n, k] for m, k, n in normalized], device=device, dtype=torch.int32)
    strides = torch.tensor(
        [
            [a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)]
            for a, b, c in zip(arrays_a, arrays_b, arrays_c)
        ],
        device=device,
        dtype=torch.int64,
    )
    metadata = [a_ptrs, b_ptrs, c_ptrs, dims, strides]
    storage_bytes = sum(item.numel() * item.element_size() for item in arrays_a + arrays_b + arrays_c + metadata)
    properties = torch.cuda.get_device_properties(device)
    sm_count = int(properties.multi_processor_count)
    device_signature = (int(properties.major) * 10 + int(properties.minor)) * 1000 + sm_count
    homogeneous = all(shape == normalized[0] for shape in normalized) and all(
        item.is_contiguous() for item in arrays_a + arrays_b + arrays_c
    )
    # TODO(H-series/SM90+): add the H-series-specific TMA, warp-specialized,
    # persistent-pipeline implementation and tune it independently. Until then
    # H-series devices intentionally use the portable path below.
    return GroupedWorkspace(
        arrays_a,
        arrays_b,
        arrays_c,
        a_ptrs,
        b_ptrs,
        c_ptrs,
        dims,
        strides,
        normalized,
        config,
        _shape_signature(normalized),
        device_signature,
        sm_count,
        storage_bytes,
        homogeneous,
    )


def build_fused_gate_up_workspace(
    shapes: Sequence[tuple[int, int, int]],
    dtype,
    seed: int = 0,
    config: KernelConfig | None = None,
) -> GroupedWorkspace:
    """Build packed W13 problems for a one-launch Gate+Up grouped GEMM.

    Every problem's output dimension is ``2 * local_ffn`` and is laid out as
    ``[gate, up]``.  Packing is represented directly by one contiguous B and C
    matrix per expert rather than by two kernel launches.  SiLU and the
    elementwise Gate/Up multiply intentionally remain outside this compute-only
    GEMM benchmark.
    """
    normalized = tuple(tuple(int(value) for value in shape) for shape in shapes)
    if any(len(shape) != 3 or shape[2] % 2 for shape in normalized):
        raise ValueError("fused Gate+Up shapes require an even packed output dimension")
    workspace = build_grouped_workspace(normalized, dtype, seed, config)
    workspace.operation = "gate_up"
    return workspace


def _grid_size(workspace: GroupedWorkspace, config: KernelConfig) -> int:
    tiles = workspace.tile_count_for(config)
    return max(1, min(tiles, workspace.sm_count * config.cta_multiplier))


def _selected_config(kernel, fallback: KernelConfig) -> KernelConfig:
    best = getattr(kernel, "best_config", None)
    if best is None:
        return fallback
    values = best.kwargs
    return KernelConfig(
        int(values["BLOCK_M"]),
        int(values["BLOCK_N"]),
        int(values["BLOCK_K"]),
        int(values["GROUP_M"]),
        int(best.num_warps),
        int(best.num_stages),
        int(values["CTA_MULTIPLIER"]),
    )


def launch_grouped(
    workspace: GroupedWorkspace,
    input_precision: str = "ieee",
    config: KernelConfig | None = None,
    *,
    autotune: bool = True,
) -> None:
    """Launch the persistent grouped kernel, optionally with a fixed config."""
    if _homogeneous_grouped_kernel_impl is None or _generic_grouped_kernel_impl is None:
        raise RuntimeError("Triton with an active CUDA driver is required")
    cfg = config or workspace.config
    if workspace.problem_count == 1:
        launch_matmul(
            workspace.a[0],
            workspace.b[0],
            workspace.c[0],
            cfg,
            input_precision,
            autotune=autotune,
        )
        workspace.config = last_matmul_config(cfg) if autotune else cfg
        workspace.scheduler = "single_problem_matmul"
        return
    common = {
        "MAX_M": workspace.max_m,
        "MAX_N": workspace.max_n,
        "MAX_K": workspace.max_k,
        "SHAPE_SIGNATURE": workspace.shape_signature,
        "DEVICE_SIGNATURE": workspace.device_signature,
        "SM_COUNT": workspace.sm_count,
        "INPUT_PRECISION": input_precision,
    }
    anchors = (workspace.a[0], workspace.b[0], workspace.c[0])

    if workspace.homogeneous:
        m, k, n = workspace.shapes[0]
        if autotune:

            def grid(meta):
                return (
                    max(
                        1,
                        min(
                            sum(
                                math.ceil(mm / meta["BLOCK_M"]) * math.ceil(nn / meta["BLOCK_N"])
                                for mm, _, nn in workspace.shapes
                            ),
                            workspace.sm_count * meta["CTA_MULTIPLIER"],
                        ),
                    ),
                )

            _homogeneous_grouped_kernel[grid](
                workspace.a_ptrs,
                workspace.b_ptrs,
                workspace.c_ptrs,
                *anchors,
                M=m,
                N=n,
                K=k,
                PROBLEM_COUNT=workspace.problem_count,
                **common,
            )
            workspace.config = _selected_config(_homogeneous_grouped_kernel, cfg)
            workspace.scheduler = "homogeneous_persistent"
            return
        _homogeneous_grouped_kernel_impl[(_grid_size(workspace, cfg),)](
            workspace.a_ptrs,
            workspace.b_ptrs,
            workspace.c_ptrs,
            *anchors,
            M=m,
            N=n,
            K=k,
            MAX_M=m,
            MAX_N=n,
            MAX_K=k,
            PROBLEM_COUNT=workspace.problem_count,
            SHAPE_SIGNATURE=workspace.shape_signature,
            DEVICE_SIGNATURE=workspace.device_signature,
            SM_COUNT=workspace.sm_count,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            GROUP_M=cfg.group_size_m,
            CTA_MULTIPLIER=cfg.cta_multiplier,
            INPUT_PRECISION=input_precision,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        workspace.config = cfg
        workspace.scheduler = "homogeneous_persistent"
        return

    if autotune:

        def grid(meta):
            return (
                max(
                    1,
                    min(
                        sum(
                            math.ceil(m / meta["BLOCK_M"]) * math.ceil(n / meta["BLOCK_N"])
                            for m, _, n in workspace.shapes
                        ),
                        workspace.sm_count * meta["CTA_MULTIPLIER"],
                    ),
                ),
            )

        _generic_grouped_kernel[grid](
            workspace.a_ptrs,
            workspace.b_ptrs,
            workspace.c_ptrs,
            workspace.dims,
            workspace.strides,
            *anchors,
            workspace.problem_count,
            **common,
        )
        workspace.config = _selected_config(_generic_grouped_kernel, cfg)
        workspace.scheduler = "generic_persistent"
        return

    _generic_grouped_kernel_impl[(_grid_size(workspace, cfg),)](
        workspace.a_ptrs,
        workspace.b_ptrs,
        workspace.c_ptrs,
        workspace.dims,
        workspace.strides,
        *anchors,
        workspace.problem_count,
        **common,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        GROUP_M=cfg.group_size_m,
        CTA_MULTIPLIER=cfg.cta_multiplier,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    workspace.config = cfg
    workspace.scheduler = "generic_persistent"
