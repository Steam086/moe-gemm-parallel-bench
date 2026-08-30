"""Reproducible validation of persisted benchmark CSV and plot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_current_rows(path: Path, run_id: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not run_id or row.get("run_id") == run_id]


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _falsey(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no"}


def _finite_positive(value: str) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number > 0.0
    except (OverflowError, TypeError, ValueError):
        return False


def _positive_integer(value: str) -> bool:
    try:
        number = int(float(value))
        return number > 0 and float(value) == number
    except (OverflowError, TypeError, ValueError):
        return False


def _moe_execution_contract_valid(row: dict[str, str]) -> bool:
    """Require measured MoE rows to use an explicit supported execution path."""
    mode = row.get("mode")
    if mode == "torch":
        return row.get("scheduler") == "sequential_torch_mm"
    if mode != "grouped":
        return False
    return (
        _truthy(row.get("autotune_enabled", ""))
        and row.get("autotune_status") == "selected"
        and row.get("config_source") in {"workload_autotune", "cold_ring_autotune"}
        and row.get("scheduler")
        in {"homogeneous_persistent", "generic_persistent", "single_problem_matmul"}
        and row.get("launches_per_iteration") == "1"
        and all(
            _positive_integer(row.get(field, ""))
            for field in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "cta_multiplier")
        )
    )


def _derivable_metrics_valid(row: dict[str, str]) -> bool:
    """Recompute FLOPs and throughput instead of trusting persisted flags."""
    try:
        m, k, n = (int(float(row[name])) for name in ("M", "K", "N"))
        num_gemms = int(float(row.get("num_gemms", "1")))
        total_flops = float(row["total_flops"])
        latency_ms = float(row["latency_ms"])
        tflops = float(row["tflops"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_flops = 2 * m * k * n * num_gemms
    expected_tflops = expected_flops / (latency_ms * 1e9) if latency_ms > 0 else math.nan
    return (
        m > 0
        and k > 0
        and n > 0
        and num_gemms > 0
        and math.isclose(total_flops, expected_flops, rel_tol=1e-12, abs_tol=0.5)
        and math.isclose(tflops, expected_tflops, rel_tol=1e-6, abs_tol=1e-12)
    )


def _projection_contract_valid(row: dict[str, str], filename: str) -> bool:
    """Verify persisted W13 Gate+Up and W2 Down semantics."""
    if filename == "moe_w1.csv":
        return (
            row.get("projection") == "W1"
            and row.get("operation") == "gate_up"
            and row.get("fused_projections") == "2"
            and row.get("output_layout") == "gate_then_up"
            and _falsey(row.get("activation_in_timed_region", ""))
        )
    if filename == "moe_w2.csv":
        return (
            row.get("projection") == "W2"
            and row.get("operation") == "down"
            and row.get("fused_projections") == "1"
            and row.get("output_layout") == "down"
            and _falsey(row.get("activation_in_timed_region", ""))
        )
    return True


def _matched_moe_flops_equal(rows: list[dict[str, str]]) -> bool:
    totals: dict[tuple[str, ...], dict[str, int]] = {}
    fields = (
        "M",
        "mode",
        "provider",
        "cache_mode",
        "dtype",
        "parallel_size",
        "num_experts",
        "topk",
        "T",
    )
    try:
        for row in rows:
            if row.get("status") != "ok":
                continue
            key = tuple(row.get(field, "") for field in fields)
            totals.setdefault(key, {})[row["parallel_type"]] = int(float(row["total_flops"]))
    except (KeyError, TypeError, ValueError):
        return False
    return all(values.get("EP") == values.get("TP") for values in totals.values() if "EP" in values and "TP" in values)


def validate_results(results_dir: str | Path = "results", plots_dir: str | Path = "plots") -> dict[str, Any]:
    results = Path(results_dir)
    plots = Path(plots_dir)
    environment_path = results / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    run_id = str(environment.get("run_id") or "")
    requested = environment.get("requested_parameters") or {}
    plots_requested = not bool(requested.get("no_plots", False))
    experiment = str(requested.get("experiment") or "all")
    selected = {"shapes", "moe", "heatmap"} if experiment == "all" else {experiment}
    files = {
        "shapes": ["gemm_shape_sweep.csv"],
        "moe": ["moe_w1.csv", "moe_w2.csv"],
        "heatmap": ["gemm_heatmap.csv"],
    }
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "selected_experiments": sorted(selected),
        "environment_present": environment_path.exists(),
        "files": {},
        "plots": {},
        "issues": [],
    }
    if not run_id:
        report["issues"].append("environment.json has no run_id")
    any_ok = False
    ok_files: set[str] = set()
    for group in sorted(selected):
        for filename in files[group]:
            rows = _read_current_rows(results / filename, run_id)
            ok_rows = [row for row in rows if row.get("status") == "ok"]
            skipped = [row for row in rows if row.get("status") == "skipped"]
            bad = [row for row in rows if row.get("status") in {"invalid", "error"}]
            numeric_ok = all(
                _finite_positive(row.get("latency_ms", ""))
                and _finite_positive(row.get("total_flops", ""))
                and _finite_positive(row.get("tflops", ""))
                for row in ok_rows
            )
            correctness_ok = all(_truthy(row.get("correct", "")) for row in ok_rows)
            metrics_ok = all(_derivable_metrics_valid(row) for row in ok_rows)
            projection_ok = all(_projection_contract_valid(row, filename) for row in ok_rows)
            flops_ok = all(
                filename not in {"moe_w1.csv", "moe_w2.csv"} or _truthy(row.get("flops_equal", "")) for row in ok_rows
            ) and (filename not in {"moe_w1.csv", "moe_w2.csv"} or _matched_moe_flops_equal(ok_rows))
            execution_ok = filename not in {"moe_w1.csv", "moe_w2.csv"} or all(
                _moe_execution_contract_valid(row) for row in ok_rows
            )
            entry = {
                "rows_for_current_run": len(rows),
                "ok": len(ok_rows),
                "skipped": len(skipped),
                "invalid_or_error": len(bad),
                "all_correct": correctness_ok,
                "positive_finite_metrics": numeric_ok,
                "derivable_metrics_recomputed": metrics_ok,
                "projection_contract": projection_ok,
                "flops_equal": flops_ok,
                "execution_contract": execution_ok,
            }
            report["files"][filename] = entry
            any_ok = any_ok or bool(ok_rows)
            if ok_rows:
                ok_files.add(filename)
            if not rows:
                report["issues"].append(f"{filename}: no rows for run_id={run_id}")
            if bad:
                report["issues"].append(f"{filename}: {len(bad)} invalid/error rows")
            if not numeric_ok or not metrics_ok or not projection_ok or not correctness_ok or not flops_ok or not execution_ok:
                report["issues"].append(f"{filename}: failed numeric/correctness/FLOP/execution checks")

    plot_files = sorted(plots.glob("figure_*.png"))
    unexpected_pdfs = sorted(plots.glob("figure_*.pdf"))
    signatures_ok = True
    expected_plot_count = 0
    if plots_requested:
        if unexpected_pdfs:
            report["issues"].append(f"unexpected PDF plot artifacts: {len(unexpected_pdfs)}")
        for path in plot_files:
            if path.stat().st_size == 0 or path.read_bytes()[:4] != b"\x89PNG":
                signatures_ok = False
                report["issues"].append(f"invalid plot artifact: {path}")
        if any_ok:
            expected_plot_count = sum(
                {
                    "gemm_shape_sweep.csv": 3,
                    "moe_w1.csv": 3,
                    "moe_w2.csv": 3,
                    "gemm_heatmap.csv": 2,
                }[filename]
                for filename in ok_files
            )
            if len(plot_files) != expected_plot_count:
                report["issues"].append(f"expected {expected_plot_count} current plot files, found {len(plot_files)}")
    report["plots"] = {
        "requested": plots_requested,
        "count": len(plot_files) if plots_requested else 0,
        "expected_count": expected_plot_count,
        "format": "png",
        "unexpected_pdf_count": len(unexpected_pdfs),
        "nonempty_and_decodable_signature": signatures_ok,
    }
    report["valid"] = not report["issues"]
    output = results / "validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    report = validate_results(args.results_dir, args.plots_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
