"""Masked Triton GEMM with shape-family autotuning and fixed-config launches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # CPU-only inspection remains possible
    torch = None
    triton = None
    tl = None


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    group_size_m: int
    num_warps: int
    num_stages: int
    # Only grouped persistent kernels use this field. Keeping it in the shared
    # record makes selected configs and benchmark provenance unambiguous.
    cta_multiplier: int = 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def select_config(m: int, n: int, k: int) -> KernelConfig:
    """Portable fallback used before an autotuned launch has selected its winner."""
    if m <= 16:
        return KernelConfig(16, 32 if n <= 32 else 64 if n < 128 else 128, 32, 4, 2, 2)
    if m <= 32:
        return KernelConfig(32, 32 if n <= 32 else 64 if n < 128 else 128, 32, 4, 2, 3)
    if m <= 64:
        return KernelConfig(64, 32 if n <= 32 else 64 if n <= 64 else 128, 32, 8, 4, 3)
    if n <= 64:
        return KernelConfig(128, 32 if n <= 32 else 64, 32, 8, 4, 3)
    return KernelConfig(64, 128, 32 if k < 8192 else 64, 8, 4, 3)


# The union deliberately includes skinny 2-warp tiles, N=32/256, K=128,
# M=256, and multiple GROUP_M choices. The candidate function selects a
# bounded shape-family subset, so each shape does not blindly benchmark all of
# them.
# TODO(H-series/SM90+): add a separately tuned TMA/warp-specialized persistent
# single-GEMM path. This change intentionally keeps the portable kernel.
_MATMUL_CONFIG_POOL: tuple[KernelConfig, ...] = (
    KernelConfig(16, 32, 32, 1, 2, 2),
    KernelConfig(16, 64, 32, 4, 2, 2),
    KernelConfig(16, 128, 32, 4, 4, 2),
    KernelConfig(16, 128, 64, 8, 4, 3),
    KernelConfig(32, 32, 32, 4, 2, 2),
    KernelConfig(32, 64, 32, 8, 2, 3),
    KernelConfig(32, 64, 64, 4, 4, 3),
    KernelConfig(32, 128, 32, 8, 4, 3),
    KernelConfig(32, 128, 64, 4, 4, 3),
    KernelConfig(32, 256, 32, 8, 8, 3),
    KernelConfig(64, 32, 32, 8, 4, 3),
    KernelConfig(64, 64, 32, 8, 4, 3),
    KernelConfig(64, 64, 64, 4, 4, 4),
    KernelConfig(64, 64, 128, 4, 4, 4),
    KernelConfig(64, 128, 32, 8, 4, 3),
    KernelConfig(64, 128, 64, 8, 4, 4),
    KernelConfig(64, 128, 128, 4, 8, 4),
    KernelConfig(64, 256, 32, 8, 8, 3),
    KernelConfig(64, 256, 64, 8, 8, 4),
    KernelConfig(128, 32, 32, 8, 4, 3),
    KernelConfig(128, 64, 32, 8, 4, 3),
    KernelConfig(128, 64, 64, 8, 4, 4),
    KernelConfig(128, 128, 32, 8, 8, 3),
    KernelConfig(128, 128, 64, 8, 8, 4),
    KernelConfig(128, 256, 64, 8, 8, 4),
    KernelConfig(256, 32, 32, 8, 8, 3),
    KernelConfig(256, 64, 32, 8, 8, 3),
    KernelConfig(256, 64, 64, 8, 8, 4),
    KernelConfig(256, 128, 64, 8, 8, 4),
)


def _family_limits(m: int, n: int) -> tuple[set[int], set[int]]:
    if m <= 16:
        block_ms = {16, 32}
    elif m <= 32:
        block_ms = {16, 32, 64}
    elif m <= 64:
        block_ms = {32, 64, 128}
    elif m <= 128:
        block_ms = {64, 128}
    else:
        block_ms = {64, 128, 256}

    if n <= 32:
        block_ns = {32, 64}
    elif n <= 64:
        block_ns = {32, 64, 128}
    elif n <= 128:
        block_ns = {64, 128, 256}
    else:
        block_ns = {128, 256}
    return block_ms, block_ns


def matmul_candidate_configs(m: int, n: int, k: int, limit: int = 10) -> tuple[KernelConfig, ...]:
    """Return a small candidate set tailored to one M/N/K shape family."""
    block_ms, block_ns = _family_limits(m, n)
    candidates = [
        cfg
        for cfg in _MATMUL_CONFIG_POOL
        if cfg.block_m in block_ms and cfg.block_n in block_ns and (cfg.block_k <= k or cfg.block_k == 32)
    ]
    fallback = select_config(m, n, k)
    if fallback not in candidates:
        candidates.append(fallback)

    def rank(cfg: KernelConfig) -> tuple[float, int, int]:
        padded_m = ((m + cfg.block_m - 1) // cfg.block_m) * cfg.block_m
        padded_n = ((n + cfg.block_n - 1) // cfg.block_n) * cfg.block_n
        output_efficiency = (m * n) / max(1, padded_m * padded_n)
        k_penalty = 0.0 if k >= cfg.block_k * 2 else 0.15
        return (-(output_efficiency - k_penalty), cfg.block_m * cfg.block_n, cfg.block_k)

    ordered = sorted(dict.fromkeys(candidates), key=rank)
    selected = ordered[: max(1, limit)]
    low_resource = next((cfg for cfg in ordered if cfg.num_warps == 2), None)
    if low_resource is not None and low_resource not in selected:
        selected[-1] = low_resource
    return tuple(selected)


def _runtime_ready() -> bool:
    """Autotuner construction needs an active driver in Triton 3.2."""
    if triton is None or torch is None or not torch.cuda.is_available():
        return False
    try:
        triton.runtime.driver.active.get_current_target()
        return True
    except Exception:
        return False


_matmul_kernel_impl = None
_matmul_kernel = None

if _runtime_ready():

    def _prune_matmul_configs(configs, named_args, **kwargs):
        arguments = {**named_args, **kwargs}
        wanted = {
            (cfg.block_m, cfg.block_n, cfg.block_k, cfg.group_size_m, cfg.num_warps, cfg.num_stages)
            for cfg in matmul_candidate_configs(int(arguments["m"]), int(arguments["n"]), int(arguments["k"]))
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
            )
            in wanted
        ]

    _triton_configs = [
        triton.Config(
            {
                "BLOCK_M": cfg.block_m,
                "BLOCK_N": cfg.block_n,
                "BLOCK_K": cfg.block_k,
                "GROUP_M": cfg.group_size_m,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        for cfg in _MATMUL_CONFIG_POOL
    ]

    @triton.jit
    def _matmul_kernel_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        DEVICE_SIGNATURE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(m, BLOCK_M)
        grid_n = tl.cdiv(n, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        first_m = group_id * GROUP_M
        group_m = tl.minimum(grid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % group_m)
        pid_n = (pid % width) // group_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
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

    _matmul_kernel = triton.autotune(
        configs=_triton_configs,
        key=[
            "m",
            "n",
            "k",
            "stride_am",
            "stride_ak",
            "stride_bk",
            "stride_bn",
            "stride_cm",
            "stride_cn",
            "DEVICE_SIGNATURE",
            "INPUT_PRECISION",
        ],
        prune_configs_by={"early_config_prune": _prune_matmul_configs},
    )(_matmul_kernel_impl)


def _device_signature(device) -> int:
    properties = torch.cuda.get_device_properties(device)
    return (int(properties.major) * 10 + int(properties.minor)) * 1000 + int(properties.multi_processor_count)


def last_matmul_config(fallback: KernelConfig) -> KernelConfig:
    """Return the configuration selected by the most recent autotuned launch."""
    if _matmul_kernel is None:
        return fallback
    best = getattr(_matmul_kernel, "best_config", None)
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
    )


def _validate_matrices(a, b, c) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("A, B, and C must be matrices")
    m, k = a.shape
    kb, n = b.shape
    if k != kb or c.shape != (m, n):
        raise ValueError(f"shape mismatch: A={a.shape}, B={b.shape}, C={c.shape}")
    if a.device != b.device or a.device != c.device or not a.is_cuda:
        raise ValueError("A, B, and C must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise ValueError("A, B, and C must have the same dtype")
    return int(m), int(n), int(k)


def launch_matmul(
    a,
    b,
    c,
    config: KernelConfig | None = None,
    input_precision: str = "ieee",
    *,
    autotune: bool = True,
) -> None:
    """Launch an autotuned GEMM, or an explicit config when autotune is false."""
    if _matmul_kernel_impl is None or (autotune and _matmul_kernel is None):
        raise RuntimeError("Triton with an active CUDA driver is required")
    m, n, k = _validate_matrices(a, b, c)
    device_signature = _device_signature(a.device)
    if autotune:

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

        _matmul_kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            DEVICE_SIGNATURE=device_signature,
            INPUT_PRECISION=input_precision,
        )
        return

    cfg = config or select_config(m, n, k)
    grid = (triton.cdiv(m, cfg.block_m) * triton.cdiv(n, cfg.block_n),)
    _matmul_kernel_impl[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        DEVICE_SIGNATURE=device_signature,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        GROUP_M=cfg.group_size_m,
        INPUT_PRECISION=input_precision,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
