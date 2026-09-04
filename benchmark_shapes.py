"""Experiment A and the two-dimensional single-GEMM efficiency sweep."""

from __future__ import annotations

import json
from typing import Any

from kernels.matmul import (
    last_matmul_config,
    launch_matmul,
    matmul_candidate_configs,
    select_config,
)
from utils.benchmark import (
    TIMING_METHOD,
    assert_close,
    available_budget,
    dtype_size,
    time_cuda,
    torch_dtype,
    tune_rotating_configs,
)
from utils.io import write_rows
from utils.metrics import arithmetic_intensity, gemm_flops, tile_metrics


def _base(args, env: dict[str, Any], experiment: str) -> dict[str, Any]:
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
        "projection": "",
        "mode": "single",
        "provider": "triton",
        "parallel_type": "",
        "parallel_size": args.parallel_size,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "T": "",
        "num_gemms": 1,
        "active_gemms": 1,
        "correct": False,
        "cache_mode": args.cache_mode,
        "model_config": args.model_config,
        "model_type": args.model_defaults.model_type,
        "requested_warmup": args.warmup,
        "requested_repeat": args.repeat,
        "requested_parameters": json.dumps(args.requested_parameters, sort_keys=True),
        "effective_parameters": json.dumps(args.effective_parameters, sort_keys=True),
        "status": "error",
    }


def _estimate(m: int, k: int, n: int, item_size: int, workspaces: int) -> int:
    return item_size * (m * k + k * n + m * n) * workspaces


def _run_one(args, env, m: int, k: int, n: int, experiment: str) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    cfg = select_config(m, n, k)
    metrics = tile_metrics([(m, k, n)], cfg.block_m, cfg.block_n, cfg.block_k, int(env.get("sm_count") or 0))
    base = _base(args, env, experiment)
    base.update(
        {
            "M": m,
            "K": k,
            "N": n,
            "total_flops": gemm_flops(m, k, n),
            "arithmetic_intensity": arithmetic_intensity(m, k, n, dtype_size(args.dtype)),
            **{f"block_{c}": getattr(cfg, f"block_{c}") for c in ("m", "n", "k")},
            "group_size_m": cfg.group_size_m,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            **metrics,
            "launches_per_iteration": 1,
            "autotune_enabled": True,
            "autotune_status": "not_run",
            "config_source": "heuristic_fallback",
        }
    )
    # Tensors from the previous case may only have become cache-reclaimable
    # after that call returned. Flush them before admitting this case.
    torch.cuda.empty_cache()
    budget = available_budget(args.memory_fraction)
    requested_ring = 1 if args.cache_mode == "hot" else max(2, args.cold_buffers)
    per_workspace = _estimate(m, k, n, dtype_size(args.dtype), 1)
    ring = min(requested_ring, max(1, budget // max(per_workspace, 1)))
    estimate = per_workspace * ring
    base.update(
        estimated_memory_bytes=estimate, memory_budget_bytes=budget, workspace_count=ring, workspace_bytes=estimate
    )
    if per_workspace > budget or (args.max_matrix_bytes and per_workspace > args.max_matrix_bytes):
        reason = "estimated memory exceeds budget" if per_workspace > budget else "shape exceeds --max-matrix-bytes"
        base.update(status="skipped", skip_reason=reason, error="")
        return [base]
    if args.cache_mode == "cold" and ring < 2:
        base.update(status="skipped", skip_reason="rotating-cold needs at least two resident workspaces", error="")
        return [base]
    workspaces: list[tuple[Any, Any, Any]] = []
    minimum_ring = 2 if args.cache_mode == "cold" else 1
    dtype = torch_dtype(args.dtype)
    # Estimates are conservative but allocator fragmentation can still cause OOM.
    # Retry with fewer rotating buffers without changing M/K/N.
    while ring >= minimum_ring:
        try:
            generator = torch.Generator(device="cuda").manual_seed(args.seed + m + n)
            workspaces = []
            for _ in range(ring):
                a = torch.empty((m, k), device="cuda", dtype=dtype).normal_(generator=generator).mul_(0.1)
                b = torch.empty((k, n), device="cuda", dtype=dtype).normal_(generator=generator).mul_(0.1)
                c = torch.empty((m, n), device="cuda", dtype=dtype)
                workspaces.append((a, b, c))
            break
        except torch.OutOfMemoryError:
            workspaces.clear()
            torch.cuda.empty_cache()
            ring -= 1
    if not workspaces:
        base.update(
            status="skipped",
            correct=False,
            error="",
            workspace_count=0,
            workspace_bytes=0,
            skip_reason="CUDA OOM while allocating the minimum resident workspace count",
        )
        return [base]
    actual_bytes = sum(x.numel() * x.element_size() for workspace in workspaces for x in workspace)
    base.update(workspace_count=ring, workspace_bytes=actual_bytes, estimated_memory_bytes=per_workspace * ring)
    try:
        rotating_tuning = None
        if args.cache_mode == "cold":

            def tune_call(candidate, index: int) -> None:
                aa, bb, cc = workspaces[index % ring]
                launch_matmul(
                    aa,
                    bb,
                    cc,
                    candidate,
                    args.input_precision,
                    autotune=False,
                )

            rotating_tuning = tune_rotating_configs(
                matmul_candidate_configs(m, n, k),
                tune_call,
                ring,
            )
            cfg = rotating_tuning.config
        # Every rotating workspace is correctness-gated before any timing.
        for a, b, c in workspaces:
            launch_matmul(
                a,
                b,
                c,
                cfg,
                args.input_precision,
                autotune=args.cache_mode == "hot",
            )
        torch.cuda.synchronize()
        if args.cache_mode == "hot":
            cfg = last_matmul_config(cfg)
        for a, b, c in workspaces:
            c.fill_(float("nan"))
            launch_matmul(a, b, c, cfg, args.input_precision, autotune=False)
        torch.cuda.synchronize()
        max_abs = 0.0
        max_rel = 0.0
        for a, b, c in workspaces:
            reference = torch.matmul(a, b)
            abs_error, rel_error = assert_close(c, reference, args.dtype)
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            del reference
        if args.cache_mode == "hot":
            cfg = last_matmul_config(cfg)
        base.update(
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k,
            group_size_m=cfg.group_size_m,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            autotune_status="selected",
            config_source="shape_family_autotune" if args.cache_mode == "hot" else "cold_ring_autotune",
            **tile_metrics([(m, k, n)], cfg.block_m, cfg.block_n, cfg.block_k, int(env.get("sm_count") or 0)),
        )
        if rotating_tuning is not None:
            base.update(
                autotune_candidates_tested=rotating_tuning.candidates_tested,
                autotune_candidates_failed=rotating_tuning.candidates_failed,
                autotune_selection_ms=rotating_tuning.median_ms,
            )

        def triton_call(index: int) -> None:
            aa, bb, cc = workspaces[index % ring]
            launch_matmul(
                aa,
                bb,
                cc,
                cfg,
                args.input_precision,
                autotune=args.cache_mode == "hot",
            )

        timing = time_cuda(triton_call, args.warmup, args.repeat, args.target_timing_ms, workspace_count=ring)
        tflops = gemm_flops(m, k, n) / (timing.median_ms * 1e9)
        triton_row = dict(base)
        triton_row.update(
            status="ok",
            error="",
            skip_reason="",
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
        rows.append(triton_row)
        if args.torch_baseline:
            # Validate the library provider independently; do not copy the
            # Triton provider's correctness/error fields into this row.
            for a, b, c in workspaces:
                torch.mm(a, b, out=c)
            torch.cuda.synchronize()
            torch_max_abs = 0.0
            torch_max_rel = 0.0
            for a, b, c in workspaces:
                reference = torch.matmul(a, b)
                abs_error, rel_error = assert_close(c, reference, args.dtype)
                torch_max_abs = max(torch_max_abs, abs_error)
                torch_max_rel = max(torch_max_rel, rel_error)
                del reference

            def torch_call(index: int) -> None:
                aa, bb, cc = workspaces[index % ring]
                torch.mm(aa, bb, out=cc)

            torch_timing = time_cuda(torch_call, args.warmup, args.repeat, args.target_timing_ms, workspace_count=ring)
            torch_tflops = gemm_flops(m, k, n) / (torch_timing.median_ms * 1e9)
            torch_row = dict(triton_row)
            torch_row.update(
                provider="torch",
                latency_ms=torch_timing.median_ms,
                latency_p20_ms=torch_timing.p20_ms,
                latency_p80_ms=torch_timing.p80_ms,
                tflops=torch_tflops,
                correct=True,
                max_abs_error=torch_max_abs,
                max_rel_error=torch_max_rel,
                peak_efficiency=torch_tflops / args.peak_tflops if args.peak_tflops else "",
                block_m="",
                block_n="",
                block_k="",
                group_size_m="",
                num_warps="",
                num_stages="",
                autotune_enabled=False,
                autotune_status="not_applicable",
                config_source="library",
                scheduler="torch_mm",
                warmup=torch_timing.warmup,
                repeat=torch_timing.repeat,
            )
            rows.append(torch_row)
        workspaces.clear()
        torch.cuda.empty_cache()
    except AssertionError as exc:
        torch.cuda.empty_cache()
        base.update(status="invalid", error=f"correctness failure: {exc}", skip_reason="", correct=False)
        rows.append(base)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        base.update(
            status="skipped", error="", skip_reason=f"CUDA OOM during correctness/reference: {exc}", correct=False
        )
        rows.append(base)
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        base.update(status="error", error=repr(exc), skip_reason="", correct=False)
        rows.append(base)
    return rows


def run_shape_sweep(args, env: dict[str, Any]) -> list[dict[str, Any]]:
    product = args.shape_product
    ms = args.shape_ms or [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
    rows: list[dict[str, Any]] = []
    for m in ms:
        if product % m:
            row = _base(args, env, "shapes")
            row.update(M=m, K=args.shape_k, N="", status="skipped", skip_reason="M does not divide shape product")
            rows.append(row)
            continue
        rows.extend(_run_one(args, env, m, args.shape_k, product // m, "shapes"))
    valid = [r for r in rows if r.get("status") == "ok" and r.get("provider") == "triton"]
    best = max((float(r["tflops"]) for r in valid), default=0.0)
    for row in rows:
        if row.get("status") == "ok" and best:
            row["normalized_efficiency"] = float(row["tflops"]) / best
    _annotate_best_measured(rows)
    write_rows(args.results_dir / "gemm_shape_sweep.csv", rows)
    return rows


def run_heatmap(args, env: dict[str, Any]) -> list[dict[str, Any]]:
    ms = args.heatmap_ms or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ns = args.heatmap_ns or [256, 512, 1024, 2048, 4096, 8192, 14336]
    rows: list[dict[str, Any]] = []
    for m in ms:
        for n in ns:
            rows.extend(_run_one(args, env, m, args.heatmap_k, n, "heatmap"))
    valid = [r for r in rows if r.get("status") == "ok" and r.get("provider") == "triton"]
    best = max((float(r["tflops"]) for r in valid), default=0.0)
    for row in rows:
        if row.get("status") == "ok" and best:
            row["normalized_efficiency"] = float(row["tflops"]) / best
    _annotate_best_measured(rows)
    write_rows(args.results_dir / "gemm_heatmap.csv", rows)
    return rows


def _annotate_best_measured(rows: list[dict[str, Any]], threshold: float = 0.90) -> None:
    """Compare providers only within the same exact M/K/N workload."""
    best_by_shape: dict[tuple[int, int, int], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["M"]), int(row["K"]), int(row["N"]))
        best_by_shape[key] = max(best_by_shape.get(key, 0.0), float(row["tflops"]))
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["M"]), int(row["K"]), int(row["N"]))
        ratio = float(row["tflops"]) / best_by_shape[key]
        row.update(
            best_measured_tflops=best_by_shape[key],
            best_measured_ratio=ratio,
            near_best_threshold=threshold,
            near_best=ratio >= threshold,
        )
