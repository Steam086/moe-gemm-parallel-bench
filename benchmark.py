"""CLI orchestrator for shape, MoE, and heatmap experiments."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_moe import run_moe
from benchmark_shapes import run_heatmap, run_shape_sweep
from plot_results import plot_all
from utils.benchmark import (
    TIMING_METHOD,
    configure_torch_matmul,
    dtype_supported,
    select_cuda_device,
)
from utils.config import load_model_defaults
from utils.hardware import write_environment
from utils.io import write_rows
from validate_results import validate_results

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG = Path("model_configs/deepseek-v3.json")


def _portable_path(value: str | Path) -> str:
    """Return publishable provenance without leaking an absolute local path."""
    path = Path(value).expanduser()
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _public_argument(name: str, value: Any) -> Any:
    if name in {"model_config", "results_dir", "plots_dir"} and value is not None:
        return _portable_path(value)
    return str(value) if isinstance(value, Path) else value


def int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute-only Triton GEMM benchmark for MoE EP versus TP")
    p.add_argument("--experiment", choices=["all", "shapes", "moe", "heatmap"], default="all")
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--parallel-size", type=int, default=8)
    p.add_argument("--num-experts", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--ffn-size", type=int)
    p.add_argument(
        "--tokens",
        type=int,
        help="Original pre-routing token count T; run one MoE M derived from T*topk/E instead of the M sweep",
    )
    p.add_argument("--moe-ms", type=int_list, help="Comma-separated tokens-per-expert sweep")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--target-timing-ms", type=float, default=2000.0, help="Automatic cap for expensive cases")
    p.add_argument("--cache-mode", choices=["hot", "cold"], default="hot")
    p.add_argument("--cold-buffers", type=int, default=2)
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--peak-tflops", type=float)
    p.add_argument("--input-precision", choices=["ieee", "tf32"], default="ieee")
    p.add_argument(
        "--device",
        default="cuda",
        help="Logical CUDA device (for example cuda or cuda:1); honors CUDA_VISIBLE_DEVICES",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--torch-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add torch.mm single-GEMM baselines and a torch grouped_mm MoE baseline",
    )
    p.add_argument("--shape-k", type=int, default=4096)
    p.add_argument("--shape-product", type=int, default=1024 * 1024)
    p.add_argument("--shape-ms", type=int_list)
    p.add_argument("--max-matrix-bytes", type=int, default=0)
    p.add_argument("--heatmap-k", type=int, help="Defaults to model hidden_size")
    p.add_argument("--heatmap-ms", type=int_list)
    p.add_argument("--heatmap-ns", type=int_list)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--plots-dir", type=Path, default=Path("plots"))
    p.add_argument("--no-plots", action="store_true")
    return p


def _unsupported_row(args, env: dict[str, Any], experiment: str, reason: str, projection: str = "") -> dict[str, Any]:
    return {
        "schema_version": "1.3",
        "timing_method": TIMING_METHOD,
        "run_id": args.run_id,
        "timestamp": args.timestamp,
        "gpu_name": env.get("gpu_name"),
        "cuda_version": env.get("cuda_version"),
        "torch_version": env.get("torch_version"),
        "triton_version": env.get("triton_version"),
        "dtype": args.dtype,
        "experiment": experiment,
        "projection": projection,
        "operation": "gate_up" if projection == "W1" else "down" if projection == "W2" else "",
        "fused_projections": 2 if projection == "W1" else 1 if projection == "W2" else "",
        "output_layout": "gate_then_up" if projection == "W1" else "down" if projection == "W2" else "",
        "activation_in_timed_region": False if projection else "",
        "parallel_size": args.parallel_size,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "correct": False,
        "cache_mode": args.cache_mode,
        "model_config": args.model_config,
        "model_type": args.model_defaults.model_type,
        "status": "skipped",
        "skip_reason": reason,
        "requested_parameters": json.dumps(args.requested_parameters, sort_keys=True),
        "effective_parameters": json.dumps(args.effective_parameters, sort_keys=True),
    }


def _print_moe_table(rows: list[dict[str, Any]], projection: str) -> None:
    paired: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") == "grouped" and row.get("status") == "ok":
            paired.setdefault(int(row["M"]), {})[str(row["parallel_type"])] = row
    label = "W1/W3 Gate+Up" if projection == "W1" else "W2 Down"
    print(f"\n{label} Grouped GEMM")
    print("| M_e | EP TFLOPS | TP TFLOPS | TP/EP | EP ms | TP ms |")
    print("|---:|---:|---:|---:|---:|---:|")
    for m, values in sorted(paired.items()):
        if "EP" not in values or "TP" not in values:
            continue
        ep, tp = values["EP"], values["TP"]
        print(
            f"| {m} | {ep['tflops']:.3f} | {tp['tflops']:.3f} | {tp['tflops'] / ep['tflops']:.3f} | {ep['latency_ms']:.4f} | {tp['latency_ms']:.4f} |"
        )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in (0,1]")
    if args.warmup < 0 or args.repeat <= 0 or args.target_timing_ms <= 0:
        raise ValueError("warmup must be non-negative; repeat and target timing must be positive")
    if args.cold_buffers <= 0 or args.max_matrix_bytes < 0:
        raise ValueError("cold-buffers must be positive and max-matrix-bytes non-negative")
    model_config_path = args.model_config.expanduser()
    if not model_config_path.is_absolute() and not model_config_path.exists():
        bundled_path = PROJECT_ROOT / model_config_path
        if bundled_path.exists():
            model_config_path = bundled_path
    model_config_path = model_config_path.resolve()
    defaults = load_model_defaults(model_config_path)
    requested = vars(args).copy()
    args.model_config = _portable_path(model_config_path)
    args.device = select_cuda_device(args.device)
    args.run_id = uuid.uuid4().hex[:12]
    args.timestamp = datetime.now(timezone.utc).isoformat()
    args.model_defaults = defaults
    args.hidden_size = defaults.hidden_size if args.hidden_size is None else args.hidden_size
    args.ffn_size = defaults.ffn_size if args.ffn_size is None else args.ffn_size
    args.num_experts = defaults.num_experts if args.num_experts is None else args.num_experts
    args.topk = defaults.topk if args.topk is None else args.topk
    args.heatmap_k = args.hidden_size if args.heatmap_k is None else args.heatmap_k
    positive_parameters = {
        "parallel-size": args.parallel_size,
        "num-experts": args.num_experts,
        "topk": args.topk,
        "hidden-size": args.hidden_size,
        "ffn-size": args.ffn_size,
        "shape-k": args.shape_k,
        "shape-product": args.shape_product,
        "heatmap-k": args.heatmap_k,
    }
    invalid = [name for name, value in positive_parameters.items() if value <= 0]
    if invalid:
        raise ValueError("these parameters must be positive: " + ", ".join(invalid))
    if args.tokens is not None and args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.peak_tflops is not None and args.peak_tflops <= 0:
        raise ValueError("--peak-tflops must be positive")
    if args.input_precision == "tf32" and args.dtype != "fp32":
        raise ValueError("--input-precision tf32 requires --dtype fp32")
    configure_torch_matmul(args.input_precision)
    for name in ("moe_ms", "shape_ms", "heatmap_ms", "heatmap_ns"):
        values = getattr(args, name)
        if values is not None and (not values or any(value <= 0 for value in values)):
            raise ValueError(f"--{name.replace('_', '-')} must contain positive integers")
    args.requested_parameters = {
        key: _public_argument(key, value) for key, value in requested.items() if key != "model_defaults"
    }
    args.effective_parameters = {
        "timing_method": TIMING_METHOD,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_size,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "parallel_size": args.parallel_size,
        "dtype": args.dtype,
        "config_ep_size": defaults.config_ep_size,
        "model_declared_dtype": defaults.dtype,
        "model_quantization": defaults.quantization,
        "cache_mode": args.cache_mode,
        "device": args.device,
        "torch_fp32_matmul_precision": "tf32" if args.input_precision == "tf32" else "ieee",
        "triton_autotune": True,
        "grouped_base_candidate_limit": 12,
        "hot_autotune_timing": "cuda_graph_hot_9_samples_25ms_target",
        "standalone_gemm_autotune_scope": "shape-family BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/num_warps/num_stages",
        "grouped_autotune_scope": "workload-aware BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/CTA multiplier/num_warps/num_stages",
        "torch_moe_baseline": "profiler-verified torch.nn.functional.grouped_mm; sequential fallback excluded",
        "w1_semantics": "one-launch packed W13 Gate+Up GEMM with [gate,up] output; activation excluded",
        "cold_autotune": "fixed candidates timed across the complete rotating workspace ring",
        "near_best_threshold": 0.90,
        "plot_format": "png",
    }
    random.seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    env = write_environment(
        args.results_dir / "environment.json",
        {
            "run_id": args.run_id,
            "timestamp": args.timestamp,
            "model_config": args.model_config,
            "model_type": defaults.model_type,
            "model_declared_dtype": defaults.dtype,
            "model_quantization": defaults.quantization,
            "requested_parameters": args.requested_parameters,
            "effective_parameters": args.effective_parameters,
            "note": (
                "Synthetic dense-dtype benchmark; model-specific deployed quantization "
                "and communication are not emulated."
            ),
        },
    )
    print(
        f"GPU: {env.get('gpu_name')} | CUDA: {env.get('cuda_version')} | torch: {env.get('torch_version')} | Triton: {env.get('triton_version')}"
    )
    print(
        f"CC: {env.get('compute_capability')} | VRAM: {(env.get('total_memory_bytes') or 0) / 2**30:.2f} GiB | dtype: {args.dtype}"
    )
    supported, reason = dtype_supported(args.dtype)
    selected = {"shapes", "moe", "heatmap"} if args.experiment == "all" else {args.experiment}
    moe_output: dict[str, list[dict[str, Any]]] = {"W1": [], "W2": []}
    if not supported or not env.get("triton_version"):
        reason = reason or "Triton unavailable"
        if "shapes" in selected:
            write_rows(args.results_dir / "gemm_shape_sweep.csv", [_unsupported_row(args, env, "shapes", reason)])
        if "heatmap" in selected:
            write_rows(args.results_dir / "gemm_heatmap.csv", [_unsupported_row(args, env, "heatmap", reason)])
        if "moe" in selected:
            write_rows(args.results_dir / "moe_w1.csv", [_unsupported_row(args, env, "moe", reason, "W1")])
            write_rows(args.results_dir / "moe_w2.csv", [_unsupported_row(args, env, "moe", reason, "W2")])
        print(f"GPU benchmark skipped: {reason}")
    else:
        if "shapes" in selected:
            run_shape_sweep(args, env)
        if "moe" in selected:
            moe_output = run_moe(args, env)
            _print_moe_table(moe_output["W1"], "W1")
            _print_moe_table(moe_output["W2"], "W2")
        if "heatmap" in selected:
            run_heatmap(args, env)
    if not args.no_plots:
        generated = plot_all(args.results_dir, args.plots_dir)
        print(f"Generated {len(generated)} plot files")
    validation = validate_results(args.results_dir, args.plots_dir)
    print(f"Validation: {'PASS' if validation['valid'] else 'FAIL'} ({args.results_dir / 'validation.json'})")
    print("\nResults:")
    current_names = ["environment.json", "validation.json"]
    if "shapes" in selected:
        current_names.append("gemm_shape_sweep.csv")
    if "moe" in selected:
        current_names.extend(["moe_w1.csv", "moe_w2.csv"])
    if "heatmap" in selected:
        current_names.append("gemm_heatmap.csv")
    if not args.no_plots:
        current_names.append("summary.md")
    for name in current_names:
        path = args.results_dir / name
        if path.exists():
            print(path)
    print("Plots:")
    if not args.no_plots:
        [print(path) for path in sorted(args.plots_dir.glob("figure_*.*")) if path.is_file()]
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
