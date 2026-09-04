"""Compare this project's grouped GEMM with the pinned portable Triton tutorial.

Both providers use identical resident inputs/outputs and graph timing. Tutorial
tail shapes are skipped, never padded. This is a single-device compute study.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from kernels.grouped_gemm import build_grouped_workspace, grouped_candidate_configs, launch_grouped
from kernels.tutorial_grouped_gemm import (
    TUTORIAL_KERNEL_AST_SHA256,
    TUTORIAL_SOURCE,
    TUTORIAL_SOURCE_SHA256,
    launch_tutorial,
    prepare_tutorial_workspace,
    tutorial_candidate_configs,
)
from utils.benchmark import TIMING_METHOD, available_budget, configure_torch_matmul, select_cuda_device, time_cuda
from utils.config import load_model_defaults
from utils.hardware import collect_environment
from utils.metrics import gemm_flops, moe_shapes, verify_ep_tp_flops


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,K,N") from exc
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive dimensions M,K,N")
    return shape


def comparison_cases(args, sm_count: int):
    if args.moe_config:
        model = load_model_defaults(args.moe_config)
        for m in args.moe_ms:
            for projection in ("W1", "W2"):
                verify_ep_tp_flops(projection, m, model.hidden_size, model.ffn_size, model.num_experts, args.parallel_size)
                for parallel in ("EP", "TP"):
                    yield f"{projection}_{parallel}_M{m}", moe_shapes(
                        projection, parallel, m, model.hidden_size, model.ffn_size, model.num_experts, args.parallel_size,
                    )
    elif args.shape:
        yield "custom", args.shape * args.experts
    else:
        yield "homogeneous", [(128, 256, 128)] * args.experts
        yield "heterogeneous", [(128, 128, 128), (256, 64, 256), (128, 256, 128)]
        yield "persistent_rounds", [(128, 256, 128)] * (sm_count + 1)


def _check_outputs(workspaces, references) -> float:
    import torch

    max_abs = 0.0
    for workspace, expected in zip(workspaces, references):
        for output, reference in zip(workspace.c, expected):
            torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
            max_abs = max(max_abs, (output.float() - reference.float()).abs().max().item())
    return max_abs


def _select(candidates, launch, workspaces, references):
    import torch
    from triton.runtime.errors import OutOfResources

    measurements = []
    failed = 0
    for candidate in candidates:
        try:
            for index, workspace in enumerate(workspaces):
                for output in workspace.c:
                    output.fill_(float("nan"))
                launch(candidate, index)
            torch.cuda.synchronize()
            # Every measured candidate must pass, not just the winning one.
            _check_outputs(workspaces, references)
            timing = time_cuda(lambda index: launch(candidate, index), 2, 9, 25.0, workspace_count=len(workspaces))
            measurements.append((timing.median_ms, candidate))
        except OutOfResources:
            failed += 1
    if not measurements:
        raise RuntimeError("all comparison candidates exceeded device resources")
    measurements.sort(key=lambda entry: entry[0])
    # Recheck only three finalists to reduce selection noise with bounded work.
    finalists = []
    for _, candidate in measurements[:3]:
        timing = time_cuda(lambda index: launch(candidate, index), 2, 21, 50.0, workspace_count=len(workspaces))
        finalists.append((timing.median_ms, candidate))
    _, best = min(finalists, key=lambda entry: entry[0])
    return best, {"tested": len(measurements), "resource_failures": failed, "finalists": len(finalists)}


def run_case(name, shapes, args, sm_count):
    import torch

    total = sum(gemm_flops(*shape) for shape in shapes)
    row = {"case": name, "shapes": shapes, "total_flops": total, "status": "skipped"}
    tutorial_configs = tutorial_candidate_configs(shapes, sm_count)
    if not tutorial_configs:
        return {**row, "reason": "tutorial has no masks: no full-tile configuration; no padding performed"}
    ring = 1 if args.cache_mode == "hot" else max(2, args.cold_buffers)
    # A/B/C, retained half-precision references and per-expert metadata, plus
    # temporary FP32 casts and one reference GEMM during reference construction.
    estimate = ring * sum(2 * (m*k + k*n + 2*m*n) + 96 for m, k, n in shapes)
    estimate += max(4 * (m*k + k*n + m*n) for m, k, n in shapes)
    torch.cuda.empty_cache()
    if estimate > available_budget(args.memory_fraction):
        return {**row, "reason": "comparison workspaces and references exceed memory budget"}
    workspaces = [build_grouped_workspace(shapes, torch.float16, args.seed + index) for index in range(ring)]
    tutorial_workspaces = [prepare_tutorial_workspace(workspace) for workspace in workspaces]
    references = [[(a.float() @ b.float()).half() for a, b, _ in workspace.problem_tensors()]
                  for workspace in workspaces]
    project_configs = grouped_candidate_configs(
        max(m for m, _, _ in shapes), max(n for _, _, n in shapes), max(k for _, k, _ in shapes),
        problem_count=len(shapes), sm_count=sm_count,
    )
    launches = {
        "project": lambda cfg, index: launch_grouped(workspaces[index], config=cfg, autotune=False),
        "tutorial": lambda cfg, index: launch_tutorial(tutorial_workspaces[index], cfg),
    }
    selected = {}
    searches = {}
    for provider, configs in (("project", project_configs), ("tutorial", tutorial_configs)):
        selected[provider], searches[provider] = _select(configs, launches[provider], workspaces, references)
    latencies = {provider: [] for provider in launches}
    errors = {}
    repeats = {}
    # Reverse order in the second round to reduce a fixed provider-order bias.
    for providers in (("project", "tutorial"), ("tutorial", "project")):
        for provider in providers:
            launch = launches[provider]
            config = selected[provider]
            for index, workspace in enumerate(workspaces):
                for output in workspace.c:
                    output.fill_(float("nan"))
                launch(config, index)
            errors[provider] = _check_outputs(workspaces, references)
            timing = time_cuda(
                lambda index: launch(config, index), args.warmup, args.repeat, args.target_timing_ms,
                workspace_count=ring,
            )
            latencies[provider].append(timing.median_ms)
            repeats[provider] = timing.repeat
    medians = {provider: statistics.median(values) for provider, values in latencies.items()}
    slowdown = medians["project"] / medians["tutorial"]
    return {
        **row, "status": "ok", "correct": True, "timing_method": TIMING_METHOD,
        "cache_mode": args.cache_mode, "workspace_count": ring,
        "configs": {provider: config.as_dict() for provider, config in selected.items()},
        "searches": searches, "latency_ms": medians, "round_latency_ms": latencies,
        "repeat": repeats, "max_abs_error": errors,
        "tflops": {provider: total / (ms * 1e9) for provider, ms in medians.items()},
        "project_over_tutorial_latency": slowdown,
        "performance_pass": args.max_slowdown is None or slowdown <= args.max_slowdown,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--shape", action="append", type=parse_shape, help="Repeat M,K,N for a heterogeneous group")
    sources.add_argument("--moe-config", type=Path, help="Compare EP/TP shapes from a model configuration")
    parser.add_argument("--experts", type=int, default=4, help="Replications of custom shapes or homogeneous default")
    parser.add_argument("--parallel-size", type=int, default=8)
    parser.add_argument("--moe-ms", type=lambda text: [int(value) for value in text.split(",")], default=[128, 256])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-mode", choices=("hot", "cold"), default="hot")
    parser.add_argument("--cold-buffers", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-timing-ms", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-slowdown", type=float, help="Optional failure threshold for project/tutorial latency")
    parser.add_argument("--output", type=Path, default=Path("results/grouped_tutorial_comparison.json"))
    args = parser.parse_args(argv)
    if (min(args.experts, args.parallel_size, args.cold_buffers, args.repeat) <= 0 or args.warmup < 0
            or not args.moe_ms or min(args.moe_ms) <= 0 or not 0 < args.memory_fraction <= 1
            or args.target_timing_ms <= 0 or (args.max_slowdown is not None and args.max_slowdown <= 0)):
        parser.error("counts, budgets and thresholds must be positive (warmup may be zero)")
    select_cuda_device(args.device)
    configure_torch_matmul("ieee")
    environment = collect_environment()
    report = {
        "environment": environment, "timing_method": TIMING_METHOD, "dtype": "fp16",
        "cache_mode": args.cache_mode, "seed": args.seed, "max_slowdown": args.max_slowdown,
        "requested_warmup": args.warmup, "requested_repeat": args.repeat,
        "target_timing_ms": args.target_timing_ms, "requested_cold_buffers": args.cold_buffers,
        "memory_fraction": args.memory_fraction, "parallel_size": args.parallel_size,
        "model_config": args.moe_config.name if args.moe_config else None,
        "tutorial_source": TUTORIAL_SOURCE, "tutorial_source_sha256": TUTORIAL_SOURCE_SHA256,
        "tutorial_kernel_ast_sha256": TUTORIAL_KERNEL_AST_SHA256,
        "scope": "portable non-TMA tutorial; shared buffers; full tiles; device-relative grids; bounded search",
        "cases": [],
    }
    if not environment["gpu_available"]:
        report["cases"].append({"status": "skipped", "reason": "CUDA unavailable"})
    else:
        import torch
        from triton.errors import TritonError

        for name, shapes in comparison_cases(args, environment["sm_count"]):
            try:
                row = run_case(name, shapes, args, environment["sm_count"])
            except torch.OutOfMemoryError:
                row = {"case": name, "status": "skipped", "reason": "CUDA OOM; requested shapes unchanged"}
            except AssertionError as exc:
                row = {"case": name, "status": "invalid", "reason": str(exc)}
            except (RuntimeError, TritonError) as exc:
                row = {"case": name, "status": "error", "reason": str(exc)}
            report["cases"].append(row)
            torch.cuda.empty_cache()
            print(f"{name}: {row['status']}" + (
                f" project/tutorial latency={row['project_over_tutorial_latency']:.3f}" if row["status"] == "ok" else ""
            ))
    measured = [row for row in report["cases"] if row["status"] == "ok"]
    failed = any(row["status"] in {"error", "invalid"} or row.get("performance_pass") is False for row in report["cases"])
    report["status"] = "failed" if failed else "passed" if measured else "skipped"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Comparison: {report['status']} ({args.output})")
    return 2 if failed else 0 if measured else 3


if __name__ == "__main__":
    raise SystemExit(main())
