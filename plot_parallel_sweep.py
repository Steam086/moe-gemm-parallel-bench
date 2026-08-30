"""Aggregate independent EP=TP runs and plot scaling across parallel sizes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

MOE_MODES = {"grouped", "torch"}
PROJECTION_LABELS = {"W1": "W1/W3 Gate+Up", "W2": "W2 Down"}

CONSISTENT_EFFECTIVE_PARAMETERS = (
    "hidden_size",
    "ffn_size",
    "num_experts",
    "topk",
    "dtype",
    "cache_mode",
)
CONSISTENT_REQUESTED_PARAMETERS = (
    "model_config",
    "dtype",
    "tokens",
    "moe_ms",
    "warmup",
    "repeat",
    "target_timing_ms",
    "cache_mode",
    "cold_buffers",
    "memory_fraction",
    "peak_tflops",
    "input_precision",
    "seed",
    "torch_baseline",
)


def _display_path(path: Path) -> str:
    """Keep aggregate provenance useful without publishing absolute host paths."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _model_label(environment: dict) -> str:
    """Derive a display label uniformly from persisted model provenance."""
    config = str(environment.get("model_config") or "")
    return Path(config).stem or str(environment.get("model_type") or "Model")


def parse_run_spec(value: str) -> tuple[int, Path]:
    """Parse PARALLEL_SIZE=RESULTS_DIR for repeatable CLI overrides."""
    size_text, separator, directory = value.partition("=")
    if not separator or not directory:
        raise argparse.ArgumentTypeError("run must use PARALLEL_SIZE=RESULTS_DIR")
    try:
        size = int(size_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("parallel size must be an integer") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("parallel size must be positive")
    return size, Path(directory)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runs(run_dirs: dict[int, Path]):
    import pandas as pd

    frames = []
    sources: dict[int, dict] = {}
    reference_parameters: dict | None = None
    reference_requested: dict | None = None
    reference_hardware: tuple | None = None
    issues: list[str] = []
    for parallel_size, results_dir in sorted(run_dirs.items()):
        environment_path = results_dir / "environment.json"
        validation_path = results_dir / "validation.json"
        if not environment_path.exists():
            issues.append(f"P={parallel_size}: missing {environment_path}")
            continue
        environment = _read_json(environment_path)
        validation = _read_json(validation_path) if validation_path.exists() else {}
        run_id = str(environment.get("run_id") or "")
        effective = environment.get("effective_parameters") or {}
        requested = environment.get("requested_parameters") or {}
        recorded_size = effective.get("parallel_size")
        if recorded_size is not None and int(recorded_size) != parallel_size:
            issues.append(f"P={parallel_size}: environment records parallel_size={recorded_size}")
        comparable = {key: effective.get(key) for key in CONSISTENT_EFFECTIVE_PARAMETERS}
        if reference_parameters is None:
            reference_parameters = comparable
        elif comparable != reference_parameters:
            issues.append(f"P={parallel_size}: effective parameters differ from the other runs: {comparable}")
        comparable_requested = {key: requested.get(key) for key in CONSISTENT_REQUESTED_PARAMETERS}
        if reference_requested is None:
            reference_requested = comparable_requested
        elif comparable_requested != reference_requested:
            issues.append(
                f"P={parallel_size}: requested benchmark controls differ from the other runs: {comparable_requested}"
            )
        gpu_name = str(environment.get("gpu_name") or "")
        hardware = tuple(
            environment.get(key)
            for key in (
                "gpu_name",
                "compute_capability",
                "cuda_version",
                "torch_version",
                "triton_version",
            )
        )
        if reference_hardware is None:
            reference_hardware = hardware
        elif hardware != reference_hardware:
            issues.append(f"P={parallel_size}: hardware/software signature differs: {hardware}")
        if not validation.get("valid", False):
            issues.append(f"P={parallel_size}: source validation did not pass")

        source_frames = []
        for projection in ("W1", "W2"):
            csv_path = results_dir / f"moe_{projection.lower()}.csv"
            if not csv_path.exists():
                issues.append(f"P={parallel_size}: missing {csv_path}")
                continue
            frame = pd.read_csv(csv_path)
            if run_id and "run_id" in frame:
                frame = frame[frame.run_id.astype(str) == run_id].copy()
            if "mode" in frame:
                # Do not reintroduce the removed per-expert single mode when
                # aggregating older result directories.
                frame = frame[frame["mode"].isin(MOE_MODES)].copy()
            if frame.empty:
                issues.append(f"P={parallel_size}: no {projection} rows for run_id={run_id}")
                continue
            recorded = pd.to_numeric(frame["parallel_size"], errors="coerce")
            if not bool(recorded.eq(parallel_size).all()):
                issues.append(f"P={parallel_size}: {projection} CSV has another parallel_size")
            frame["source_results_dir"] = _display_path(results_dir)
            source_frames.append(frame)
            frames.append(frame)

        source = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
        sources[parallel_size] = {
            "results_dir": _display_path(results_dir),
            "run_id": run_id,
            "gpu_name": gpu_name,
            "model_name": _model_label(environment),
            "effective_parameters": comparable,
            "validation_valid": bool(validation.get("valid", False)),
            "rows": len(source),
            "ok": int((source.status == "ok").sum()) if "status" in source else 0,
            "skipped": int((source.status == "skipped").sum()) if "status" in source else 0,
        }
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, sources, issues


def _numeric(frame):
    import pandas as pd

    frame = frame.copy()
    for column in (
        "parallel_size",
        "M",
        "T",
        "tflops",
        "latency_ms",
        "total_flops",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _paired(ok):
    ep = ok[ok.parallel_type == "EP"]
    tp = ok[ok.parallel_type == "TP"]
    keys = [
        "projection",
        "parallel_size",
        "M",
        "mode",
        "cache_mode",
        "dtype",
        "num_experts",
        "topk",
        "T",
    ]
    keys = [key for key in keys if key in ok.columns]
    return ep.merge(tp, on=keys, suffixes=("_ep", "_tp"), validate="one_to_one").assign(
        tflops_ratio=lambda x: x.tflops_tp / x.tflops_ep,
        latency_ratio=lambda x: x.latency_ms_tp / x.latency_ms_ep,
    )


def _m_values(frame) -> list[int]:
    return sorted({int(value) for value in frame.M.dropna().unique()})


def _colors(ms: Iterable[int]):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = list(ms)
    norm = LogNorm(vmin=min(values), vmax=max(values)) if len(values) > 1 else None
    cmap = plt.get_cmap("viridis")
    return {m: cmap(norm(m) if norm else 0.5) for m in values}


def _format_parallel_axis(ax, parallel_sizes: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(parallel_sizes, [str(size) for size in parallel_sizes])
    ax.set_xlabel("parallel size P (EP=P or TP=P)")
    ax.grid(alpha=0.3)


def _legend(fig, axes, title: str = "M_e") -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title=title,
            loc="center left",
            bbox_to_anchor=(0.985, 0.5),
            frameon=False,
        )


def _save(fig, output: Path, stem: str) -> Path:
    import matplotlib.pyplot as plt

    path = output / f"{stem}.png"
    fig.tight_layout(rect=(0, 0, 0.91, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _absolute_plot(ok, output: Path, projection: str, value: str, model_name: str) -> Path | None:
    import matplotlib.pyplot as plt

    data = ok[(ok.projection == projection) & (ok["mode"] == "grouped")]
    if data.empty:
        return None
    ms = _m_values(data)
    colors = _colors(ms)
    sizes = sorted({int(value) for value in data.parallel_size.unique()})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, parallel_type in zip(axes, ("EP", "TP")):
        part = data[data.parallel_type == parallel_type]
        for m in ms:
            group = part[part.M == m].sort_values("parallel_size")
            if not group.empty:
                ax.plot(
                    group.parallel_size,
                    group[value],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=colors[m],
                    label=str(m),
                )
        ax.set_title(parallel_type)
        _format_parallel_axis(ax, sizes)
        ax.set_yscale("log", base=2)
    ylabel = "TFLOPS" if value == "tflops" else "latency (ms)"
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{model_name} MoE {PROJECTION_LABELS[projection]}: grouped {ylabel} across parallel sizes")
    _legend(fig, axes)
    suffix = "tflops" if value == "tflops" else "latency"
    number = {("W1", "tflops"): 1, ("W2", "tflops"): 2, ("W1", "latency_ms"): 5, ("W2", "latency_ms"): 6}[
        (projection, value)
    ]
    return _save(fig, output, f"figure_parallel_{number}_{projection.lower()}_{suffix}")


def _ratio_plot(paired, output: Path, projection: str, model_name: str) -> Path | None:
    import matplotlib.pyplot as plt

    data = paired[paired.projection == projection]
    if data.empty:
        return None
    ms = _m_values(data)
    colors = _colors(ms)
    sizes = sorted({int(value) for value in data.parallel_size.unique()})
    modes = [mode for mode in ("grouped", "torch") if mode in set(data["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        part = data[data["mode"] == mode]
        for m in ms:
            group = part[part.M == m].sort_values("parallel_size")
            if not group.empty:
                ax.plot(
                    group.parallel_size,
                    group.tflops_ratio,
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=colors[m],
                    label=str(m),
                )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(mode)
        _format_parallel_axis(ax, sizes)
    axes[0].set_ylabel("TP TFLOPS / EP TFLOPS")
    fig.suptitle(
        f"{model_name} MoE {PROJECTION_LABELS[projection]}: equal-FLOP TP/EP ratio across parallel sizes"
    )
    _legend(fig, axes)
    number = 3 if projection == "W1" else 4
    return _save(fig, output, f"figure_parallel_{number}_{projection.lower()}_tp_ep_ratio")


def _summary(path: Path, sources: dict[int, dict], paired, combined, model_name: str, parameters: dict) -> None:
    controls = ", ".join(
        f"{key}={parameters.get(key)}"
        for key in ("dtype", "cache_mode", "num_experts", "topk", "hidden_size", "ffn_size")
    )
    lines = [
        f"# {model_name} parallel-size sweep",
        "",
        "Each point compares matching `EP=P` and `TP=P` compute-only workloads with equal useful FLOPs per rank.",
        f"Validated common controls: {controls}.",
        "",
        "## Source runs",
        "",
        "| P | run_id | valid | ok rows | skipped rows | results directory |",
        "|---:|---|:---:|---:|---:|---|",
    ]
    for size, source in sorted(sources.items()):
        lines.append(
            f"| {size} | `{source['run_id']}` | {'yes' if source['validation_valid'] else 'no'} | "
            f"{source['ok']} | {source['skipped']} | `{source['results_dir']}` |"
        )
    lines += [
        "",
        "## Grouped-kernel TP/EP throughput ratios",
        "",
        "| projection | P | matched M_e points | min | median | max | worst M_e |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = paired[paired["mode"] == "grouped"]
    for (projection, size), part in grouped.groupby(["projection", "parallel_size"], sort=True):
        worst = part.loc[part.tflops_ratio.idxmin()]
        lines.append(
            f"| {PROJECTION_LABELS.get(projection, projection)} | {int(size)} | {len(part)} | "
            f"{part.tflops_ratio.min():.3f} | "
            f"{part.tflops_ratio.median():.3f} | {part.tflops_ratio.max():.3f} | {int(worst.M)} |"
        )
    grouped_tp = combined[(combined["mode"] == "grouped") & (combined.parallel_type == "TP")].copy()
    gate_up = grouped_tp[grouped_tp.projection == "W1"]
    down = grouped_tp[grouped_tp.projection == "W2"]
    latency = gate_up.merge(
        down,
        on=["parallel_size", "M", "mode", "parallel_type"],
        how="outer",
        suffixes=("_gate_up", "_down"),
        validate="one_to_one",
    )
    lines += [
        "",
        "## TP grouped-kernel projection latency",
        "",
        "Gate+Up is one packed W13 Triton launch. Combined time is the sum of the two compute-only launches; "
        "SiLU×Up and communication are not included.",
        "",
        "| P | M_e | Gate+Up ms/status | Down ms/status | combined GEMM ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in latency.sort_values(["parallel_size", "M"]).iterrows():
        gate_up_ok = str(row.status_gate_up) == "ok"
        down_ok = str(row.status_down) == "ok"
        gate_up_ms = float(row.latency_ms_gate_up) if gate_up_ok else None
        down_ms = float(row.latency_ms_down) if down_ok else None
        gate_up_display = f"{gate_up_ms:.4f}" if gate_up_ms is not None else "skipped"
        down_display = f"{down_ms:.4f}" if down_ms is not None else "skipped"
        combined_display = f"{gate_up_ms + down_ms:.4f}" if gate_up_ms is not None and down_ms is not None else "—"
        lines.append(
            f"| {int(row.parallel_size)} | {int(row.M)} | {gate_up_display} | "
            f"{down_display} | {combined_display} |"
        )
    lines += [
        "",
        "Skipped source rows are retained in the aggregate CSV with their recorded reasons; plots do not interpolate them.",
        "Only GEMM compute is timed; communication and routing costs are excluded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_and_plot(
    run_dirs: dict[int, Path],
    output_csv: Path,
    plots_dir: Path,
    summary_path: Path,
    validation_path: Path,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    combined, sources, issues = _load_runs(run_dirs)
    if combined.empty:
        raise ValueError("no source rows were loaded")
    first_source = next(iter(sources.values()))
    model_name = str(first_source["model_name"])
    parameters = dict(first_source["effective_parameters"])
    combined = _numeric(combined)
    combined.sort_values(
        ["projection", "parallel_size", "M", "mode", "parallel_type"],
        inplace=True,
        kind="stable",
    )
    combined.to_csv(output_csv, index=False)
    ok = combined[combined.status == "ok"].copy()
    bad = combined[combined.status.isin(["invalid", "error"])]
    if not bad.empty:
        issues.append(f"aggregate contains {len(bad)} invalid/error rows")
    if not bool(ok.correct.astype(str).str.lower().isin(["true", "1"]).all()):
        issues.append("one or more status=ok rows failed correctness")
    if not bool(ok.flops_equal.astype(str).str.lower().isin(["true", "1"]).all()):
        issues.append("one or more status=ok rows failed equal-FLOP validation")
    paired = _paired(ok)
    if paired.empty:
        issues.append("no matched status=ok EP/TP pairs were found")
    unpaired_ok_rows = len(ok) - 2 * len(paired)

    for stale in plots_dir.glob("figure_parallel_*.*"):
        if stale.suffix in {".png", ".pdf"}:
            stale.unlink()
    figures = []
    for projection in ("W1", "W2"):
        for value in ("tflops",):
            figure = _absolute_plot(ok, plots_dir, projection, value, model_name)
            if figure:
                figures.append(figure)
        figure = _ratio_plot(paired, plots_dir, projection, model_name)
        if figure:
            figures.append(figure)
    for projection in ("W1", "W2"):
        figure = _absolute_plot(ok, plots_dir, projection, "latency_ms", model_name)
        if figure:
            figures.append(figure)
    figures.sort()
    if len(figures) != 6:
        issues.append(f"expected 6 aggregate figures, generated {len(figures)}")
    for figure in figures:
        if figure.stat().st_size == 0 or figure.read_bytes()[:4] != b"\x89PNG":
            issues.append(f"invalid PNG: {figure}")

    _summary(summary_path, sources, paired, combined, model_name, parameters)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid": not issues,
        "parallel_sizes": sorted(run_dirs),
        "source_runs": sources,
        "aggregate_csv": _display_path(output_csv),
        "rows": len(combined),
        "ok": len(ok),
        "skipped": int((combined.status == "skipped").sum()),
        "invalid_or_error": len(bad),
        "matched_ep_tp_pairs": len(paired),
        "unpaired_ok_rows": unpaired_ok_rows,
        "figures": [_display_path(path) for path in figures],
        "issues": issues,
    }
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        help="repeatable PARALLEL_SIZE=RESULTS_DIR (at least two are required)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/parallel_sweep/moe_parallel_sweep.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/parallel_sweep/summary.md"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/parallel_sweep/validation.json"),
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("plots/parallel_sweep"))
    args = parser.parse_args()
    run_dirs = dict(args.run)
    if len(run_dirs) < 2:
        parser.error("at least two distinct parallel-size runs are required")
    report = aggregate_and_plot(
        run_dirs,
        args.output_csv,
        args.plots_dir,
        args.summary,
        args.validation,
    )
    print(
        f"Aggregated {report['rows']} rows across P={report['parallel_sizes']}; "
        f"generated {len(report['figures'])} figures"
    )
    print(f"Validation: {'PASS' if report['valid'] else 'FAIL'} ({args.validation})")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
