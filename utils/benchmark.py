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
TIMING_METHOD = "cuda_graph_events"


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


def _whole_ring_count(requested: int, minimum: int, workspace_count: int) -> int:
    """Round upward so even short measurements visit the entire cold ring."""
    return math.ceil(max(minimum, requested) / workspace_count) * workspace_count


def _capture_samples(function: Callable[[int], None], count: int, workspace_count: int, stream):
    """Capture event nodes around calls, not around host-side graph submission.

    External events become explicit graph nodes. A replay therefore executes
    every start/kernel/end sequence on the device with no Python between them.
    Inputs and library allocator storage must already be warmed on ``stream``.
    """
    import torch

    events = [
        (torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True))
        for _ in range(count)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for index, (start, end) in enumerate(events):
            start.record()
            function(index % workspace_count)
            end.record()
    return graph, events


def _replay_samples(graph, events, stream) -> list[float]:
    import torch

    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def time_cuda(
    function: Callable[[int], None], requested_warmup: int, requested_repeat: int, target_ms: float = 2000.0,
    *, workspace_count: int = 1,
) -> TimingResult:
    """Measure device execution inside one CUDA Graph replay.

    Capture, Python dispatch, compilation, allocation and warmup are untimed.
    Each sample is bounded by event nodes *inside* the graph, so CPU submission
    starvation cannot inflate it. Cold samples traverse complete workspace rings.
    Capture failures propagate; there is no eager timing fallback.
    """
    import torch

    if workspace_count < 1 or requested_warmup < 0 or requested_repeat < 1 or target_ms <= 0:
        raise ValueError("invalid CUDA graph timing counts or target")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(workspace_count):
            function(index)  # compile/autotune and allocator warmup outside capture
    stream.synchronize()
    pilot, pilot_events = _capture_samples(function, workspace_count, workspace_count, stream)
    pilot_ms = max(statistics.median(_replay_samples(pilot, pilot_events, stream)), 0.001)
    warmup = max(2, min(requested_warmup, int(max(2.0, 250.0 / pilot_ms))))
    repeat = max(3, min(requested_repeat, int(max(3.0, target_ms / pilot_ms))))
    warmup = _whole_ring_count(warmup, 2, workspace_count)
    repeat = _whole_ring_count(repeat, 3, workspace_count)
    with torch.cuda.stream(stream):
        for _ in range(warmup // workspace_count):
            pilot.replay()
    stream.synchronize()
    del pilot, pilot_events
    graph, events = _capture_samples(function, repeat, workspace_count, stream)
    # Warm the instantiated graph and its private allocator pool before sampling.
    _replay_samples(graph, events, stream)
    samples = _replay_samples(graph, events, stream)
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
            timing = time_cuda(
                lambda index: launch(config, index), 2, sample_count, workspace_count=workspace_count,
            )
            candidate_median = timing.median_ms
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


def hot_autotune_bench(function: Callable[[], None], quantiles):
    """Bounded hot candidate timing, using the same device-only graph method.

    Triton's default do_bench clears L2 before each sample. That selects for a
    different cache regime and spends ~100 ms per candidate. Nine samples with
    a 25 ms target bound measurement work; compilation remains uncapped.
    """
    timing = time_cuda(lambda _: function(), 2, 9, target_ms=25.0)
    values = {0.2: timing.p20_ms, 0.5: timing.median_ms, 0.8: timing.p80_ms}
    return [values[quantile] for quantile in quantiles]


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
