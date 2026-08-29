"""Persistent grouped GEMM over distinct expert A/B/C pointers.

Equal, contiguous MoE problems use a constexpr-shape fast path. Arbitrary
shape/stride groups use a device-side scheduled fallback with a per-problem K
loop, so a short-K problem never executes dot operations up to the group's
maximum K.
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
    matmul_candidate_configs,
    select_config,
)


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


def grouped_candidate_configs(
    max_m: int,
    max_n: int,
    max_k: int,
    base_limit: int = 6,
) -> tuple[KernelConfig, ...]:
    """Shape-family grouped candidates including persistent CTA-count tuning."""
    base = matmul_candidate_configs(max_m, max_n, max_k, limit=base_limit)
    candidates: list[KernelConfig] = []
    for cfg in base:
        multipliers = (1, 2, 4) if cfg.block_m <= 32 else (1, 2)
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
        for cfg in _MATMUL_CONFIG_POOL
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
                int(arguments["MAX_M"]), int(arguments["MAX_N"]), int(arguments["MAX_K"])
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        CTA_MULTIPLIER: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        grid_size = tl.num_programs(0) + CTA_MULTIPLIER * 0
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
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                a = tl.load(
                    a_ptr + offs_m[:, None] * K + k_start + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (k_start + offs_k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :],
                    mask=(k_start + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                )
                accumulator += tl.dot(a, b, input_precision=INPUT_PRECISION)
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        CTA_MULTIPLIER: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        grid_size = tl.num_programs(0) + CTA_MULTIPLIER * 0
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
    common = {
        "MAX_M": workspace.max_m,
        "MAX_N": workspace.max_n,
        "MAX_K": workspace.max_k,
        "SHAPE_SIGNATURE": workspace.shape_signature,
        "DEVICE_SIGNATURE": workspace.device_signature,
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
