"""Recreate all figures and the data-driven Markdown summary from persisted CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MOE_MODES = {"grouped", "torch"}
PROJECTION_LABELS = {"W1": "W1/W3 Gate+Up", "W2": "W2 Down"}


def _supported_moe_modes(frame):
    if frame.empty or "mode" not in frame:
        return frame
    # Old result directories can contain the removed per-expert `single` mode
    # and sequential torch baseline. Replotting must not resurrect either.
    supported = frame[frame["mode"].isin(MOE_MODES)].copy()
    if "scheduler" not in supported:
        return supported[supported["mode"] != "torch"].copy()
    return supported[
        (supported["mode"] != "torch") | (supported["scheduler"] == "torch_grouped_mm")
    ].copy()


def _load(path: Path, run_id: str | None = None):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if run_id and "run_id" in frame:
        frame = frame[frame.run_id.astype(str) == run_id].copy()
    if "status" in frame:
        frame = frame[frame.status == "ok"].copy()
    for column in (
        "M",
        "K",
        "N",
        "tflops",
        "normalized_efficiency",
        "arithmetic_intensity",
        "latency_ms",
        "total_output_tiles_per_rank",
        "estimated_waves",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _model_label(environment: dict) -> str:
    """Derive a display label uniformly from persisted model provenance."""
    config = str(environment.get("model_config") or "")
    return Path(config).stem or str(environment.get("model_type") or "Model")


def _save(fig, plots: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(plots / f"{stem}.png", dpi=180, bbox_inches="tight")


def _line_shape(frame, plots: Path, y: str, ylabel: str, stem: str, title: str) -> None:
    import matplotlib.pyplot as plt

    if frame.empty or "provider" not in frame:
        return
    data = frame[frame.provider == "triton"].sort_values("M", ascending=False)
    if data.empty:
        return
    labels = [f"M={int(m)},N={int(n)}" for m, n in zip(data.M, data.N)]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(len(data)), data[y], marker="o")
    ax.set_xticks(range(len(data)), labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_title(title + f" (K={int(data.K.iloc[0])}; M*N*K constant)")
    _save(fig, plots, stem)
    plt.close(fig)


def _moe_curves(frame, plots: Path, projection: str, model_name: str) -> None:
    import matplotlib.pyplot as plt

    frame = _supported_moe_modes(frame)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, parallel), group in frame.groupby(["mode", "parallel_type"]):
        group = group.sort_values("M")
        ax.plot(group.M, group.tflops, marker="o", label=f"{parallel} {mode}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens per expert M_e")
    ax.set_ylabel("TFLOPS")
    ax.set_title(f"{model_name} MoE {PROJECTION_LABELS[projection]}: EP vs TP compute-only")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, plots, f"figure_{'4' if projection == 'W1' else '6'}_moe_{projection.lower()}_tflops")
    plt.close(fig)


def _ratio(frame, value: str):
    ep = frame[frame.parallel_type == "EP"]
    tp = frame[frame.parallel_type == "TP"]
    candidate_keys = [
        "run_id",
        "M",
        "mode",
        "cache_mode",
        "dtype",
        "projection",
        "provider",
        "parallel_size",
        "num_experts",
        "topk",
        "T",
    ]
    keys = [key for key in candidate_keys if key in frame.columns]
    return ep.merge(tp, on=keys, suffixes=("_ep", "_tp"), validate="one_to_one").assign(
        ratio=lambda x: x[f"{value}_tp"] / x[f"{value}_ep"]
    )


def _ratio_plot(frame, plots: Path, projection: str, model_name: str, latency: bool = False) -> None:
    import matplotlib.pyplot as plt

    frame = _supported_moe_modes(frame)
    value = "latency_ms" if latency else "tflops"
    data = _ratio(frame, value)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in data.groupby("mode"):
        group = group.sort_values("M")
        ax.plot(group.M, group.ratio, marker="o", label=mode)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens per expert M_e")
    ax.set_ylabel("TP latency / EP latency" if latency else "TP TFLOPS / EP TFLOPS")
    ax.set_title(
        f"{model_name} MoE {PROJECTION_LABELS[projection]} TP/EP "
        f"{'latency' if latency else 'performance'} ratio"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    number = "8" if latency else ("5" if projection == "W1" else "7")
    _save(fig, plots, f"figure_{number}_moe_{projection.lower()}_{'latency_' if latency else ''}ratio")
    plt.close(fig)


def _heatmap(frame, plots: Path, normalized: bool, model_name: str) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    if "provider" in frame:
        frame = frame[frame.provider == "triton"]
    if frame.empty:
        return
    value = "normalized_efficiency" if normalized else "tflops"
    pivot = frame.pivot_table(index="M", columns="N", values=value, aggfunc="median").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    ax.set_xlabel("N (K fixed)")
    ax.set_ylabel("M")
    metric_title = "Normalized GEMM efficiency heatmap" if normalized else "GEMM TFLOPS heatmap"
    ax.set_title(f"{model_name}: {metric_title}")
    fig.colorbar(image, ax=ax, label="TFLOPS / max" if normalized else "TFLOPS")
    _save(fig, plots, f"figure_9_heatmap{'_normalized' if normalized else ''}")
    plt.close(fig)


def generate_summary(results: Path, run_id: str | None = None, model_name: str = "Model") -> Path:
    import pandas as pd

    shapes = _load(results / "gemm_shape_sweep.csv", run_id)
    w1 = _supported_moe_modes(_load(results / "moe_w1.csv", run_id))
    w2 = _supported_moe_modes(_load(results / "moe_w2.csv", run_id))
    lines = [
        f"# {model_name} benchmark summary",
        "",
        f"Run ID: `{run_id or 'unavailable'}`.",
        "",
        "This report is generated from persisted `status=ok` rows for this invocation; it does not assume TP is slower.",
        "",
    ]
    triton_shapes = shapes[shapes.provider == "triton"] if not shapes.empty else shapes
    if not triton_shapes.empty:
        hi, lo = triton_shapes.tflops.max(), triton_shapes.tflops.min()
        best = triton_shapes.loc[triton_shapes.tflops.idxmax()]
        small = triton_shapes.sort_values("M")
        threshold_rows = small[small.tflops < 0.7 * hi]
        threshold = int(threshold_rows.M.max()) if not threshold_rows.empty else None
        aspect = min(float(best.M), float(best.N)) / max(float(best.M), float(best.N))
        central = triton_shapes[(triton_shapes.M >= 128) & (triton_shapes.M <= 4096)]
        central_floor = central.tflops.min() / hi if not central.empty else float("nan")
        lines += [
            "## Fixed-MNK shape sweep",
            "",
            f"1. Observed throughput spans **{lo:.3f}–{hi:.3f} TFLOPS**, a **{hi / lo:.2f}×** range (max/min).",
            f"2. The best observed shape is M={int(best.M)}, N={int(best.N)}, K={int(best.K)} (M/N aspect metric {aspect:.3f}). Shapes with M=128–4096 remain at least {central_floor:.1%} of peak, while the most extreme small-M cases fall much lower; the data therefore favor the broad non-extreme plateau, not necessarily the exactly square point.",
            f"3. Throughput falls below 70% of the observed maximum by approximately M≤{threshold}."
            if threshold
            else "3. No measured M fell below 70% of the observed maximum.",
            "",
        ]
    else:
        lines += ["## Fixed-MNK shape sweep", "", "No valid GPU rows were recorded.", ""]
    lines += ["## MoE EP versus TP (equal useful FLOPs per rank)", ""]
    for projection, frame in (("W1", w1), ("W2", w2)):
        projection_label = PROJECTION_LABELS[projection]
        if frame.empty:
            lines += [f"### {projection_label}", "", "No valid matched GPU rows were recorded.", ""]
            continue
        ratio = _ratio(frame, "tflops")
        lines += [f"### {projection_label}", ""]
        for mode in ("grouped", "torch"):
            part = ratio[ratio["mode"] == mode]
            if part.empty:
                lines.append(f"- {mode}: no matched EP/TP rows.")
            else:
                lines.append(
                    f"- {mode}: TP/EP throughput ratio {part.ratio.min():.3f}–{part.ratio.max():.3f} (median {part.ratio.median():.3f})."
                )
        grouped = ratio[ratio["mode"] == "grouped"].sort_values("M")
        if not grouped.empty:
            near = [str(int(x)) for x in grouped.loc[grouped.ratio.between(0.9, 1.1), "M"]]
            slow = [str(int(x)) for x in grouped.loc[grouped.ratio < 0.9, "M"]]
            lines.append(
                f"- Grouped TP≈EP (ratio 0.9–1.1) at M_e={','.join(near) or 'none'}; TP is >10% slower at M_e={','.join(slow) or 'none'}."
            )
            lines += [
                "",
                "| M_e | grouped EP TFLOPS | grouped TP TFLOPS | TP/EP | EP ms | TP ms |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
            for _, item in grouped.iterrows():
                lines.append(
                    f"| {int(item.M)} | {item.tflops_ep:.3f} | {item.tflops_tp:.3f} | {item.ratio:.3f} | {item.latency_ms_ep:.4f} | {item.latency_ms_tp:.4f} |"
                )
        lines.append("")
    ratio_frames = [
        part[part["mode"] == "grouped"]
        for part in (
            _ratio(w1, "tflops") if not w1.empty else pd.DataFrame(),
            _ratio(w2, "tflops") if not w2.empty else pd.DataFrame(),
        )
        if not part.empty and not part[part["mode"] == "grouped"].empty
    ]
    all_ratios = pd.concat(ratio_frames, ignore_index=True) if ratio_frames else pd.DataFrame()
    material_count = int(((all_ratios.ratio < 0.9) | (all_ratios.ratio > 1.1)).sum()) if not all_ratios.empty else 0
    near_parity_note = ""
    if not w1.empty:
        w1_grouped = _ratio(w1, "tflops")
        w1_grouped = w1_grouped[w1_grouped["mode"] == "grouped"]
        if not w1_grouped.empty and bool(w1_grouped.ratio.between(0.9, 1.1).any()):
            near_parity_note = (
                " At least one grouped W1 point is near parity, showing that a shape difference does not "
                "require a slowdown when aggregate scheduling compensates."
            )
    conclusion = (
        f"Yes. {material_count}/{len(all_ratios)} matched grouped M/projection comparisons differ from parity by more than 10%; equal FLOPs coexist with materially different throughput.{near_parity_note}"
        if material_count
        else "No material >10% matched throughput difference was observed in this run."
    )
    correlations = []
    for projection, frame in (("W1", w1), ("W2", w2)):
        grouped_frame = frame[frame["mode"] == "grouped"]
        if len(grouped_frame) >= 3:
            correlations.append(
                f"{PROJECTION_LABELS[projection]}: corr(TFLOPS, AI)="
                f"{grouped_frame.tflops.corr(grouped_frame.arithmetic_intensity):.3f}, "
                f"corr(TFLOPS, output tiles)="
                f"{grouped_frame.tflops.corr(grouped_frame.total_output_tiles_per_rank):.3f}, "
                f"corr(TFLOPS, waves)={grouped_frame.tflops.corr(grouped_frame.estimated_waves):.3f}"
            )
    lines += [
        "## Interpretation",
        "",
        "4–7. Exact per-M EP/TP TFLOPS and latency are in the CSVs and Figures 4–8; grouped is the Triton MoE kernel under study and torch is the optional native grouped_mm library baseline.",
        "8. Treat ratios near 1 (roughly 0.9–1.1) as similar; inspect Figures 5/7 for the measured M ranges.",
        "9. Simple Pearson relationships (descriptive, not causal): "
        + ("; ".join(correlations) if correlations else "insufficient valid rows."),
        "10. Does this run support ‘equal theoretical FLOPs do not guarantee equal throughput’? **" + conclusion + "**",
        "",
        "## Scope and caveats",
        "",
        f"Only GEMM compute is timed. W1 is one packed W13 Gate+Up launch with [gate,up] output; SiLU and the elementwise multiply are excluded. Routing, permutation, All-to-All, AllReduce/ReduceScatter, and network communication are excluded. Rotating-cold buffers reduce reuse but do not prove every access misses L2. Arithmetic intensity is an ideal one-read/one-write model, not measured HBM traffic. {model_name} dimensions are used with synthetic dense FP16/BF16/FP32 data; deployed quantization is not emulated.",
        "",
    ]
    path = results / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_all(results_dir: str | Path = "results", plots_dir: str | Path = "plots") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    results, plots = Path(results_dir), Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    environment_path = results / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    run_id = str(environment.get("run_id") or "") or None
    model_name = _model_label(environment)
    # A selective rerun must not leave old figures looking like current output.
    for stale in plots.glob("figure_*.*"):
        if stale.suffix in {".png", ".pdf"}:
            stale.unlink()
    shapes = _load(results / "gemm_shape_sweep.csv", run_id)
    _line_shape(
        shapes, plots, "tflops", "TFLOPS", "figure_1_shape_tflops", f"{model_name}: Fixed-FLOP GEMM shape throughput"
    )
    _line_shape(
        shapes,
        plots,
        "normalized_efficiency",
        "TFLOPS / max observed",
        "figure_2_shape_normalized_efficiency",
        f"{model_name}: Fixed-FLOP normalized efficiency",
    )
    _line_shape(
        shapes,
        plots,
        "arithmetic_intensity",
        "ideal FLOPs / byte",
        "figure_3_arithmetic_intensity",
        f"{model_name}: Idealized arithmetic intensity",
    )
    w1, w2 = _load(results / "moe_w1.csv", run_id), _load(results / "moe_w2.csv", run_id)
    _moe_curves(w1, plots, "W1", model_name)
    _ratio_plot(w1, plots, "W1", model_name)
    _moe_curves(w2, plots, "W2", model_name)
    _ratio_plot(w2, plots, "W2", model_name)
    _ratio_plot(w1, plots, "W1", model_name, True)
    _ratio_plot(w2, plots, "W2", model_name, True)
    heatmap = _load(results / "gemm_heatmap.csv", run_id)
    _heatmap(heatmap, plots, False, model_name)
    _heatmap(heatmap, plots, True, model_name)
    generate_summary(results, run_id, model_name)
    return sorted(plots.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()
    paths = plot_all(args.results_dir, args.plots_dir)
    print(f"Generated {len(paths)} plot files and {Path(args.results_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
