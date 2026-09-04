"""PyTorch native grouped-GEMM workspace for homogeneous MoE experts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate

try:
    import torch
except ImportError:
    torch = None


class TorchGroupedMMUnavailable(RuntimeError):
    """Raised when the public PyTorch grouped-GEMM API cannot run."""


@dataclass
class TorchGroupedWorkspace:
    mat_a: object
    mat_b: object
    offs: object
    shapes: tuple[tuple[int, int, int], ...]
    output: object | None = None
    operation: str = "gemm"
    scheduler: str = "torch_grouped_mm"

    @property
    def problem_count(self) -> int:
        return len(self.shapes)

    @property
    def storage_bytes(self) -> int:
        tensors = (self.mat_a, self.mat_b, self.offs, self.output)
        return sum(item.untyped_storage().nbytes() for item in tensors if item is not None)

    def problem_tensors(self) -> Iterator[tuple[object, object, object]]:
        if self.output is None:
            raise RuntimeError("torch grouped_mm has not produced an output")
        start = 0
        for index, (m, _, _) in enumerate(self.shapes):
            end = start + m
            yield self.mat_a[start:end], self.mat_b[index], self.output[start:end]
            start = end


def torch_grouped_mm_unavailable_reason(dtype=None, problem_count: int = 1) -> str:
    if torch is None:
        return "PyTorch unavailable"
    if not hasattr(torch.nn.functional, "grouped_mm"):
        return "torch.nn.functional.grouped_mm unavailable; PyTorch 2.10 or newer is required"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    # PyTorch 2.13 GroupedBlas.cpp uses sequential mm + offs.cpu() outside
    # this domain. Never equate one Python API call with one grouped kernel.
    if dtype is not None and dtype != torch.bfloat16:
        return "native grouped_mm fast path requires BF16; requested dtype would use sequential fallback"
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability[0] not in (9, 10):
        return "native grouped_mm fast path is only admitted on SM90/SM100-family devices; fallback excluded"
    if problem_count >= 1024:
        return "native grouped_mm fast path requires fewer than 1024 groups"
    return ""


def inspect_grouped_trace(cpu_ops: Sequence[str], cuda_events: Sequence[str]) -> dict[str, int | bool]:
    """Fail closed unless a trace proves one CUTLASS grouped compute kernel.

    CUDA events here are device activities, not CUDA API calls. Library metadata
    preparation kernels count as launches but not as GEMMs. Missing CUPTI data,
    new unrecognized kernels and sequential fallbacks are all unverified.
    """
    if any(name in {"aten::mm", "aten::mm_out", "aten::bmm", "aten::matmul"} for name in cpu_ops):
        raise TorchGroupedMMUnavailable("profiler detected sequential/batched matmul fallback")
    if list(cpu_ops).count("aten::_grouped_mm") != 1:
        raise TorchGroupedMMUnavailable("profiler did not observe exactly one native grouped_mm call")
    names = [name.lower() for name in cuda_events]
    if any("memcpy" in name for name in names):
        raise TorchGroupedMMUnavailable("grouped_mm probe contains a device transfer; fallback excluded")
    kernels = [name for name in names if "memset" not in name]
    compute = [
        name for name in kernels
        if "cutlass" in name and "gemm" in name and "groupproblemshape" in name
    ]
    if len(compute) != 1 or any(
        name not in compute and "prepare_grouped_gemm_data" not in name for name in kernels
    ):
        raise TorchGroupedMMUnavailable("CUDA trace does not prove one supported grouped compute kernel")
    return {
        "execution_verified": True,
        "api_calls_per_iteration": 1,
        "launches_per_iteration": len(kernels),
        "grouped_compute_launches": 1,
    }


def verify_torch_grouped_execution(workspace: TorchGroupedWorkspace) -> dict[str, int | bool]:
    """Profile this exact shape/dtype outside measurement before admitting it."""
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise TorchGroupedMMUnavailable("CUDA profiler/CUPTI unavailable; grouped execution cannot be verified")
    try:
        launch_torch_grouped(workspace)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as profile:
            launch_torch_grouped(workspace)
            torch.cuda.synchronize()
        events = profile.events()
        return inspect_grouped_trace(
            [event.name for event in events if event.device_type == torch.autograd.DeviceType.CPU],
            [event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA],
        )
    except TorchGroupedMMUnavailable:
        raise
    except RuntimeError as exc:
        raise TorchGroupedMMUnavailable(f"native grouped execution probe failed: {exc}") from exc


def build_torch_grouped_workspace(
    shapes: Sequence[tuple[int, int, int]],
    dtype,
    seed: int = 0,
    *,
    operation: str = "gemm",
) -> TorchGroupedWorkspace:
    reason = torch_grouped_mm_unavailable_reason(dtype, len(shapes))
    if reason:
        raise TorchGroupedMMUnavailable(reason)
    if not shapes:
        raise ValueError("torch grouped GEMM requires at least one problem")
    normalized = tuple(tuple(int(value) for value in shape) for shape in shapes)
    if any(len(shape) != 3 or any(value <= 0 for value in shape) for shape in normalized):
        raise ValueError("every torch grouped shape must be a positive (M, K, N) triple")
    _, shared_k, shared_n = normalized[0]
    if any(k != shared_k or n != shared_n for _, k, n in normalized):
        raise ValueError("torch grouped_mm 2D/3D mode requires shared K and N dimensions")
    group_ends = tuple(accumulate(m for m, _, _ in normalized))
    if group_ends[-1] > 2**31 - 1:
        raise ValueError("torch grouped_mm offsets exceed int32 range")

    item_size = torch.empty((), dtype=dtype).element_size()
    stride_alignment = max(1, 16 // item_size)

    def aligned(value: int) -> int:
        return (value + stride_alignment - 1) // stride_alignment * stride_alignment

    generator = torch.Generator(device="cuda").manual_seed(seed)
    # grouped_mm requires matrix row strides aligned to 16 bytes. Padding the
    # backing storage preserves the exact logical K/N dimensions and FLOPs.
    mat_a = torch.empty((group_ends[-1], aligned(shared_k)), device="cuda", dtype=dtype)[:, :shared_k]
    mat_b = torch.empty(
        (len(normalized), shared_k, aligned(shared_n)), device="cuda", dtype=dtype
    )[:, :, :shared_n]
    mat_a.normal_(generator=generator).mul_(0.1)
    mat_b.normal_(generator=generator).mul_(0.1)
    offs = torch.tensor(group_ends, device=mat_a.device, dtype=torch.int32)
    return TorchGroupedWorkspace(mat_a, mat_b, offs, normalized, operation=operation)


def launch_torch_grouped(workspace: TorchGroupedWorkspace) -> None:
    """Run one public ``torch.nn.functional.grouped_mm`` call.

    The public API has no ``out=`` argument. Dropping the previous result before
    dispatch lets PyTorch's warmed caching allocator reuse the same output
    storage across benchmark repetitions.
    """
    reason = torch_grouped_mm_unavailable_reason(workspace.mat_a.dtype, workspace.problem_count)
    if reason:
        raise TorchGroupedMMUnavailable(reason)
    workspace.output = None
    workspace.output = torch.nn.functional.grouped_mm(workspace.mat_a, workspace.mat_b, offs=workspace.offs)
    workspace.scheduler = "torch_grouped_mm"
