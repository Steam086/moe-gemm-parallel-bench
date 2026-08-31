"""CUDA timing, dtype, memory, and correctness helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class TimingResult:
    p20_ms: float
    median_ms: float
    p80_ms: float
    mean_ms: float
    std_ms: float
    warmup: int
    repeat: int


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class TuningResult:
    config: object
    median_ms: float
    candidates_tested: int
    candidates_failed: int


def torch_dtype(name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def dtype_size(name: str) -> int:
    return 4 if name == "fp32" else 2


def select_cuda_device(device_spec: str) -> str:
    """Select one logical CUDA device and return its canonical ``cuda:N`` name.

    ``CUDA_VISIBLE_DEVICES`` remains authoritative: indices are logical indices
    inside the visible set, never physical ``nvidia-smi`` indices.
    """
    try:
        import torch
    except ImportError:
        if device_spec != "cuda" and not device_spec.startswith("cuda:"):
            raise ValueError("--device must name a CUDA device (for example cuda or cuda:1)")
        return device_spec

    device = torch.device(device_spec)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device (for example cuda or cuda:1)")
    if not torch.cuda.is_available():
        return device_spec
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"--device selects logical CUDA device {index}, but only {torch.cuda.device_count()} device(s) are visible"
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def configure_torch_matmul(input_precision: str) -> None:
    """Make PyTorch's FP32 CUDA policy match Triton's requested dot policy."""
    try:
        import torch
    except ImportError:
        return

    allow_tf32 = input_precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    # Keep high-level matmul APIs aligned with the backend flag on PyTorch
    # versions that also consult this process-wide policy.
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def dtype_supported(name: str) -> tuple[bool, str]:
    """Probe the requested dtype with a real CUDA matmul, not capability inference alone."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA unavailable"
        if name == "bf16" and not torch.cuda.is_bf16_supported():
            return False, "BF16 unsupported by this GPU/PyTorch build"
        dtype = torch_dtype(name)
        probe = torch.ones((16, 16), device="cuda", dtype=dtype)
        result = torch.mm(probe, probe)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(result).all().item()):
            return False, f"{name} CUDA matmul produced non-finite output"
        del probe, result
        return True, ""
    except Exception as exc:
        return False, f"{name} CUDA matmul probe failed: {exc!r}"


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def time_cuda(
    function: Callable[[int], None], requested_warmup: int, requested_repeat: int, target_ms: float = 2000.0
) -> TimingResult:
    """Measure back-to-back CUDA stream execution with CUDA events.

    Indexing permits rotating resident workspaces. Callers compile, autotune,
    and warm any library-managed caching-allocator storage before measured
    samples. All measured work is enqueued before one final synchronization so
    each repetition is not forced into an artificial empty-stream
    request/response cycle.
    """
    import torch

    function(0)  # compile/autotune outside timing
    torch.cuda.synchronize()
    pilot_start, pilot_end = torch.cuda.Event(True), torch.cuda.Event(True)
    pilot_start.record()
    function(0)
    pilot_end.record()
    pilot_end.synchronize()
    pilot_ms = max(pilot_start.elapsed_time(pilot_end), 0.001)
    warmup = max(2, min(requested_warmup, int(max(2.0, 250.0 / pilot_ms))))
    repeat = max(3, min(requested_repeat, int(max(3.0, target_ms / pilot_ms))))
    for index in range(warmup):
        function(index)
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(repeat)]
    for index, (start, end) in enumerate(events):
        start.record()
        function(index)
        end.record()
    events[-1][1].synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    return TimingResult(
        p20_ms=percentile(samples, 0.2),
        median_ms=statistics.median(samples),
        p80_ms=percentile(samples, 0.8),
        mean_ms=statistics.mean(samples),
        std_ms=statistics.pstdev(samples),
        warmup=warmup,
        repeat=repeat,
    )


def tune_rotating_configs(
    configs: Sequence[ConfigT],
    launch: Callable[[ConfigT, int], None],
    workspace_count: int,
    repeats_per_workspace: int = 2,
) -> TuningResult:
    """Choose a fixed config while rotating through every resident workspace.

    Compilation and a complete ring warmup happen before candidate timing. This
    makes the selection reflect the benchmark's cold-ring access pattern rather
    than Triton's default repeated launch on one hot argument set.
    """
    import torch

    try:
        from triton.errors import TritonError
    except ImportError:
        TritonError = RuntimeError

    if not configs:
        raise ValueError("at least one tuning config is required")
    if workspace_count < 2:
        raise ValueError("rotating autotune requires at least two workspaces")
    best_config = None
    best_median = math.inf
    tested = 0
    failed = 0
    sample_count = max(3, workspace_count * max(1, repeats_per_workspace))
    for config in configs:
        try:
            launch(config, 0)  # compile outside candidate timing
            torch.cuda.synchronize()
            for index in range(workspace_count):
                launch(config, index)
            torch.cuda.synchronize()
            events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(sample_count)]
            for index, (start, end) in enumerate(events):
                start.record()
                launch(config, index % workspace_count)
                end.record()
            events[-1][1].synchronize()
            candidate_median = statistics.median(start.elapsed_time(end) for start, end in events)
            tested += 1
            if candidate_median < best_median:
                best_config = config
                best_median = candidate_median
        except (RuntimeError, AssertionError, TritonError):
            # Match Triton's autotuner behavior for compilation and resource
            # failures: discard the candidate and continue.
            failed += 1
            torch.cuda.synchronize()
    if best_config is None:
        raise RuntimeError(f"all {len(configs)} rotating-autotune candidates failed")
    return TuningResult(best_config, best_median, tested, failed)


def assert_close(actual, reference, dtype_name: str) -> tuple[float, float]:
    import torch

    diff = (actual.float() - reference.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = reference.float().abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    tolerances = {"fp16": (2e-2, 2e-2), "bf16": (5e-2, 5e-2), "fp32": (2e-3, 2e-3)}
    rtol, atol = tolerances[dtype_name]
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    return max_abs, max_rel


def available_budget(memory_fraction: float) -> int:
    import torch

    free, _ = torch.cuda.mem_get_info()
    return max(0, int(free * memory_fraction) - 256 * 1024**2)
