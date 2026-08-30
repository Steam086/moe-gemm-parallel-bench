# Plot and result interpretation guide

The repository does not commit machine-specific benchmark outputs. Run `benchmark.py` to generate CSVs and PNGs under the ignored `results/` and `plots/` directories, then use this guide to interpret them.

## Generate figures

```bash
python benchmark.py
# Or replot an existing run without GPU work:
python plot_results.py --results-dir results --plots-dir plots
```

A complete run can generate:

1. fixed-FLOP shape throughput;
2. fixed-FLOP normalized efficiency;
3. ideal arithmetic intensity;
4. packed W1/W3 Gate+Up EP/TP throughput;
5. packed W1/W3 Gate+Up TP/EP throughput ratio;
6. W2 Down EP/TP throughput;
7. W2 Down TP/EP throughput ratio;
8. Gate+Up and Down TP/EP latency ratios;
9. raw and normalized GEMM heatmaps.

Only figures supported by current-run CSV rows are emitted. Selective reruns remove stale root figures before plotting.

## Reading the axes

GEMM shapes use `[M,K] @ [K,N]`:

- `M`: rows, commonly tokens per expert (`M_e`);
- `K`: reduction dimension;
- `N`: output dimension;
- higher TFLOPS is better;
- lower latency is better.

For TP/EP throughput ratios:

```text
ratio = TP TFLOPS / EP TFLOPS
```

- `1.0`: equal observed throughput;
- above `1.0`: TP is faster;
- below `1.0`: EP is faster.

For latency ratios:

```text
ratio = TP latency / EP latency
```

- `1.0`: equal observed latency;
- below `1.0`: TP has lower latency;
- above `1.0`: EP has lower latency.

## Normalized plots

Normalized efficiency is relative to the best measured row in the same run, not hardware theoretical peak. It should be used to compare shape sensitivity inside that run only.

The ideal arithmetic-intensity plot uses:

```text
2*M*K*N / (sizeof(dtype) * (M*K + K*N + M*N))
```

It does not measure HBM or cache traffic.

## Heatmaps

Heatmaps hold `K` fixed and sweep `M` and `N`. Confirm the recorded `heatmap_k` before relating a cell to a model projection. W1 sharding changes `N`; W2 sharding changes `K`, so a W1-oriented heatmap does not directly characterize W2.

## Parallel-size plots

Aggregate explicit source runs:

```bash
python plot_parallel_sweep.py \
  --run 2=results/parallel_sweep/p2 \
  --run 4=results/parallel_sweep/p4 \
  --run 8=results/parallel_sweep/p8
```

The aggregate includes skipped rows in its CSV but does not interpolate them in plots. Read each recorded `skip_reason` instead of assuming a particular GPU-memory cause.
Its generated summary also records the exact grouped TP Gate+Up, Down, and
combined GEMM latency at every matched `(P, M_e)` point. Gate+Up is one packed
W13 GEMM launch; the SiLU×Up activation is not part of that timing.

## Reporting results

When sharing a figure, also provide:

- source CSVs and `environment.json`;
- validation report;
- model configuration and runtime overrides;
- dtype and FP32/TF32 policy;
- cache mode and workspace count;
- requested/effective warmup and repeat counts;
- selected logical CUDA device and software versions;
- a statement that routing and communication are excluded.

Inspect artifacts for local paths or other metadata before publication. Never present one machine's autotune selections or throughput as portable to another GPU.
