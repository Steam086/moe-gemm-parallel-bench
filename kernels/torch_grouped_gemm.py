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


def torch_grouped_mm_unavailable_reason() -> str:
    if torch is None:
        return "PyTorch unavailable"
    if not hasattr(torch.nn.functional, "grouped_mm"):
        return "torch.nn.functional.grouped_mm unavailable; PyTorch 2.10 or newer is required"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability < (8, 0):
        return f"torch grouped_mm requires SM >= 80; selected device is SM {capability[0]}{capability[1]}"
    return ""


def build_torch_grouped_workspace(
    shapes: Sequence[tuple[int, int, int]],
    dtype,
    seed: int = 0,
    *,
    operation: str = "gemm",
) -> TorchGroupedWorkspace:
    reason = torch_grouped_mm_unavailable_reason()
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
    reason = torch_grouped_mm_unavailable_reason()
    if reason:
        raise TorchGroupedMMUnavailable(reason)
    workspace.output = None
    workspace.output = torch.nn.functional.grouped_mm(workspace.mat_a, workspace.mat_b, offs=workspace.offs)
    workspace.scheduler = "torch_grouped_mm"
