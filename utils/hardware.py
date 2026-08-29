"""Environment discovery that remains importable without PyTorch or CUDA."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu_available": False,
        "gpu_name": "unavailable",
        "logical_cuda_device": None,
        "cuda_driver_version": None,
        "cuda_driver_api_version": None,
        "cuda_version": None,
        "nvcc_version": _command(["nvcc", "--version"]),
        "torch_version": None,
        "triton_version": None,
        "compute_capability": None,
        "total_memory_bytes": None,
        "free_memory_bytes": None,
        "sm_count": None,
        "fp16_supported": False,
        "bf16_supported": False,
        "fp32_supported": False,
        "tf32_supported": False,
        "tensor_core_capability": "unknown",
        "dtype_probes": {},
    }
    smi_banner = _command(["nvidia-smi"])
    if smi_banner:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_banner)
        if match:
            env["cuda_driver_api_version"] = match.group(1)
    driver_query = _command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if driver_query:
        # The driver version is process-wide. Device-specific provenance comes
        # from PyTorch so CUDA_VISIBLE_DEVICES remapping cannot mismatch it.
        env["cuda_driver_version"] = driver_query.splitlines()[0].strip()
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(device_index)
            free, total = torch.cuda.mem_get_info(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            major, minor = capability
            env.update(
                gpu_available=True,
                gpu_name=prop.name,
                logical_cuda_device=f"cuda:{device_index}",
                compute_capability=f"{major}.{minor}",
                total_memory_bytes=int(prop.total_memory),
                runtime_total_memory_bytes=int(total),
                free_memory_bytes=int(free),
                sm_count=int(prop.multi_processor_count),
                tf32_supported=major >= 8,
                tensor_core_capability=(
                    "Hopper fourth-generation Tensor Cores (FP16/BF16/TF32/FP8 hardware support)"
                    if major == 9
                    else "Ada fourth-generation Tensor Cores (FP16/BF16/TF32/FP8 hardware support)"
                    if major == 8 and minor == 9
                    else "Ampere third-generation Tensor Cores (FP16/BF16/TF32 hardware support)"
                    if major == 8
                    else "Tensor Cores present"
                    if major >= 7
                    else "No Tensor Cores"
                ),
            )
            probes: dict[str, dict[str, Any]] = {}
            for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
                try:
                    a = torch.ones((16, 16), device="cuda", dtype=dtype)
                    result = torch.mm(a, a)
                    torch.cuda.synchronize()
                    probes[name] = {
                        "torch_matmul_ok": True,
                        "result_dtype": str(result.dtype),
                        "finite": bool(torch.isfinite(result).all().item()),
                    }
                    del a, result
                except Exception as probe_exc:
                    probes[name] = {"torch_matmul_ok": False, "error": repr(probe_exc)}
            env["dtype_probes"] = probes
            env["fp16_supported"] = bool(probes["fp16"].get("torch_matmul_ok"))
            env["bf16_supported"] = bool(probes["bf16"].get("torch_matmul_ok"))
            env["fp32_supported"] = bool(probes["fp32"].get("torch_matmul_ok"))
    except Exception as exc:
        env["torch_probe_error"] = repr(exc)
    try:
        import triton

        env["triton_version"] = triton.__version__
    except Exception as exc:
        env["triton_probe_error"] = repr(exc)
    return env


def write_environment(path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    env = collect_environment()
    if extra:
        env.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return env
