"""Experiment B: fair per-rank EP versus TP expert GEMM workloads."""

from __future__ import annotations

import json
from typing import Any

from kernels.grouped_gemm import (
    build_grouped_workspace,
    grouped_candidate_configs,
    launch_grouped,
)
from kernels.matmul import select_config
from utils.benchmark import (
    assert_close,
    available_budget,
    dtype_size,
    time_cuda,
    torch_dtype,
    tune_rotating_configs,
)
from utils.io import write_rows
from utils.metrics import arithmetic_intensity, moe_shapes, tile_metrics, verify_ep_tp_flops


def _estimate(shapes: list[tuple[int, int, int]], item_size: int) -> int:
    data = sum(item_size * (m * k + k * n + m * n) for m, k, n in shapes)
    # Three pointer tables, dimensions, and six per-problem strides. Tile
    # assignment is computed on device and needs no host-built tile maps.
    metadata = len(shapes) * (3 * 8 + 3 * 4 + 6 * 8)
    return data + metadata


def _base(args, env, projection: str, parallel_type: str, mode: str, m: int, shape, count: int, total: int):
    k, n = shape[1], shape[2]
    cfg = select_config(m, n, k)
    metrics = tile_metrics([shape] * count, cfg.block_m, cfg.block_n, cfg.block_k, int(env.get("sm_count") or 0))
    tokens = m * args.num_experts // args.topk
    provider = "torch" if mode == "torch" else "triton"
    row = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "timestamp": args.timestamp,
        "gpu_name": env.get("gpu_name"),
        "cuda_version": env.get("cuda_version"),
        "torch_version": env.get("torch_version"),
        "triton_version": env.get("triton_version"),
        "dtype": args.dtype,
        "experiment": "moe",
        "projection": projection,
        "mode": mode,
        "provider": provider,
        "parallel_type": parallel_type,
        "parallel_size": args.parallel_size,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "T": tokens,
        "M": m,
        "K": k,
        "N": n,
        "num_gemms": count,
        "active_gemms": count,
        "total_flops": total,
        "arithmetic_intensity": arithmetic_intensity(m, k, n, dtype_size(args.dtype)),
        "correct": False,
        "cache_mode": args.cache_mode,
        "tokens_per_expert": m,
        "weight_shard_factor": 1 if parallel_type == "EP" else args.parallel_size,
        "flops_equal": True,
        "block_m": cfg.block_m,
        "block_n": cfg.block_n,
        "block_k": cfg.block_k,
        "group_size_m": cfg.group_size_m,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        **metrics,
        "launches_per_iteration": 1 if mode == "grouped" else count,
        "autotune_enabled": mode != "torch",
        "autotune_status": "not_run" if mode != "torch" else "not_applicable",
        "config_source": "heuristic_fallback" if mode != "torch" else "library",
        "requested_warmup": args.warmup,
        "requested_repeat": args.repeat,
        "model_config": args.model_config,
        "model_type": args.model_defaults.model_type,
        "requested_parameters": json.dumps(args.requested_parameters, sort_keys=True),
        "effective_parameters": json.dumps(args.effective_parameters, sort_keys=True),
        "status": "error",
    }
    if mode == "torch":
        row.update(
            block_m="",
            block_n="",
            block_k="",
            group_size_m="",
            num_warps="",
            num_stages="",
        )
    return row


def _run_case(args, env: dict[str, Any], projection: str, parallel_type: str, mode: str, m: int) -> dict[str, Any]:
    import torch

    total = verify_ep_tp_flops(projection, m, args.hidden_size, args.ffn_size, args.num_experts, args.parallel_size)
    shapes = moe_shapes(
        projection, parallel_type, m, args.hidden_size, args.ffn_size, args.num_experts, args.parallel_size
    )
    row = _base(args, env, projection, parallel_type, mode, m, shapes[0], len(shapes), total)
    # The previous case may have released its tensors after its final
    # empty_cache() call while the local workspace variable still held the
    # allocation.  Flush those now-unused cached blocks before asking the CUDA
    # driver how much memory is available for this case.  Otherwise large
    # parallel-size sweeps can incorrectly skip every case after the first.
    torch.cuda.empty_cache()
    budget = available_budget(args.memory_fraction)
    per_workspace = _estimate(shapes, dtype_size(args.dtype))
    requested_ring = 1 if args.cache_mode == "hot" else max(2, args.cold_buffers)
    ring = min(requested_ring, max(0, budget // max(per_workspace, 1)))
    if args.cache_mode == "hot":
        ring = min(1, ring)
    estimate = per_workspace * max(ring, 1)
    row.update(
        estimated_memory_bytes=estimate, memory_budget_bytes=budget, workspace_count=ring, workspace_bytes=estimate
    )
    if per_workspace > budget:
        row.update(status="skipped", skip_reason="estimated one-workspace memory exceeds budget", error="")
        return row
    if args.cache_mode == "cold" and ring < 2:
        row.update(status="skipped", skip_reason="rotating-cold needs at least two resident workspaces", error="")
        return row
    cfg = select_config(m, shapes[0][2], shapes[0][1])
    dtype = torch_dtype(args.dtype)
    workspaces: list[Any] = []
    minimum_ring = 2 if args.cache_mode == "cold" else 1
    # Retry allocator failures with fewer rotating buffers; never alter GEMM shapes.
    while ring >= minimum_ring:
        try:
            workspaces = [
                build_grouped_workspace(shapes, dtype, args.seed + index * 100003, cfg) for index in range(ring)
            ]
            break
        except torch.OutOfMemoryError:
            workspaces.clear()
            torch.cuda.empty_cache()
            ring -= 1
    if not workspaces:
        row.update(
            status="skipped",
            correct=False,
            error="",
            workspace_count=0,
            workspace_bytes=0,
            skip_reason="CUDA OOM while allocating the minimum resident workspace count",
        )
        return row
    row.update(
        workspace_count=ring,
        workspace_bytes=sum(x.storage_bytes for x in workspaces),
        estimated_memory_bytes=per_workspace * ring,
    )
    try:
        rotating_tuning = None
        if args.cache_mode == "cold" and mode != "torch":

            def tune_call(candidate, index: int) -> None:
                current = workspaces[index % ring]
                launch_grouped(
                    current,
                    args.input_precision,
                    candidate,
                    autotune=False,
                )

            candidates = grouped_candidate_configs(
                max(mm for mm, _, _ in shapes),
                max(nn for _, _, nn in shapes),
                max(kk for _, kk, _ in shapes),
                problem_count=len(shapes),
                sm_count=workspaces[0].sm_count,
            )
            rotating_tuning = tune_rotating_configs(candidates, tune_call, ring)
            cfg = rotating_tuning.config
        # Correctness-gate every workspace that the timed rotating ring can select.
        for workspace in workspaces:
            if mode == "grouped":
                launch_grouped(
                    workspace,
                    args.input_precision,
                    cfg,
                    autotune=args.cache_mode == "hot",
                )
            else:
                for a, b, c in zip(workspace.a, workspace.b, workspace.c):
                    torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        max_abs = 0.0
        max_rel = 0.0
        for workspace in workspaces:
            for a, b, c in zip(workspace.a, workspace.b, workspace.c):
                reference = torch.matmul(a, b)
                abs_error, rel_error = assert_close(c, reference, args.dtype)
                max_abs = max(max_abs, abs_error)
                max_rel = max(max_rel, rel_error)
                del reference
        if mode != "torch":
            cfg = workspaces[0].config
            row.update(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                group_size_m=cfg.group_size_m,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                cta_multiplier=cfg.cta_multiplier,
                autotune_status="selected",
                config_source=("cold_ring_autotune" if args.cache_mode == "cold" else "workload_autotune"),
                scheduler=workspaces[0].scheduler,
                **tile_metrics(
                    shapes,
                    cfg.block_m,
                    cfg.block_n,
                    cfg.block_k,
                    int(env.get("sm_count") or 0),
                ),
            )
            if rotating_tuning is not None:
                row.update(
                    autotune_candidates_tested=rotating_tuning.candidates_tested,
                    autotune_candidates_failed=rotating_tuning.candidates_failed,
                    autotune_selection_ms=rotating_tuning.median_ms,
                )
        else:
            row["scheduler"] = "sequential_torch_mm"

        def invoke(index: int) -> None:
            current = workspaces[index % ring]
            if mode == "grouped":
                launch_grouped(
                    current,
                    args.input_precision,
                    cfg,
                    autotune=args.cache_mode == "hot",
                )
            else:
                for aa, bb, cc in zip(current.a, current.b, current.c):
                    torch.mm(aa, bb, out=cc)

        timing = time_cuda(invoke, args.warmup, args.repeat, args.target_timing_ms)
        tflops = total / (timing.median_ms * 1e9)
        row.update(
            status="ok",
            skip_reason="",
            error="",
            correct=True,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            latency_ms=timing.median_ms,
            latency_p20_ms=timing.p20_ms,
            latency_p80_ms=timing.p80_ms,
            tflops=tflops,
            peak_efficiency=tflops / args.peak_tflops if args.peak_tflops else "",
            warmup=timing.warmup,
            repeat=timing.repeat,
        )
        workspaces.clear()
        torch.cuda.empty_cache()
    except AssertionError as exc:
        row.update(status="invalid", correct=False, error=f"correctness failure: {exc}", skip_reason="")
        print(f"INVALID {projection} {parallel_type} {mode} M={m}: {exc}")
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as exc:
        row.update(
            status="skipped", correct=False, error="", skip_reason=f"CUDA OOM during correctness/reference: {exc}"
        )
        print(f"SKIP {projection} {parallel_type} {mode} M={m}: CUDA OOM")
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        row.update(status="error", correct=False, error=repr(exc), skip_reason="")
        print(f"ERROR {projection} {parallel_type} {mode} M={m}: {exc}")
        torch.cuda.empty_cache()
    return row


def run_moe(args, env: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if args.num_experts % args.parallel_size:
        raise ValueError("num_experts must be divisible by parallel_size")
    if args.ffn_size % args.parallel_size:
        raise ValueError("ffn_size must be divisible by parallel_size")
    ms = args.moe_ms or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    if args.tokens is not None:
        assignments = args.tokens * args.topk
        if assignments % args.num_experts:
            raise ValueError("tokens * topk must be divisible by num_experts")
        ms = [assignments // args.num_experts]
    if any(m <= 0 for m in ms):
        raise ValueError("all tokens-per-expert M values must be positive")
    nonintegral = [m for m in ms if (m * args.num_experts) % args.topk]
    if nonintegral:
        raise ValueError(
            "M*num_experts must be divisible by topk so T is integral; invalid M values: "
            + ",".join(map(str, nonintegral))
        )
    outputs: dict[str, list[dict[str, Any]]] = {"W1": [], "W2": []}
    for projection in ("W1", "W2"):
        for m in ms:
            modes = ["grouped"]
            if args.torch_baseline:
                modes.append("torch")
            for mode in modes:
                for parallel_type in ("EP", "TP"):
                    outputs[projection].append(_run_case(args, env, projection, parallel_type, mode, m))
        valid = [r for r in outputs[projection] if r.get("status") == "ok"]
        best = max((float(r["tflops"]) for r in valid), default=0.0)
        for row in outputs[projection]:
            if row.get("status") == "ok" and best:
                row["normalized_efficiency"] = float(row["tflops"]) / best
        _annotate_best_measured(outputs[projection])
        write_rows(args.results_dir / f"moe_{projection.lower()}.csv", outputs[projection])
    return outputs


def _annotate_best_measured(rows: list[dict[str, Any]], threshold: float = 0.90) -> None:
    """Compare schedulers/providers for one M and parallel workload only."""
    best_by_case: dict[tuple[int, str], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["M"]), str(row["parallel_type"]))
        best_by_case[key] = max(best_by_case.get(key, 0.0), float(row["tflops"]))
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["M"]), str(row["parallel_type"]))
        ratio = float(row["tflops"]) / best_by_case[key]
        row.update(
            best_measured_tflops=best_by_case[key],
            best_measured_ratio=ratio,
            near_best_threshold=threshold,
            near_best=ratio >= threshold,
        )
