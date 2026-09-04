# MoE GEMM Parallel Benchmark

A compute-only CUDA/Triton benchmark for studying how MoE Tensor Parallelism (TP) and Expert Parallelism (EP) can have equal useful FLOPs but different GEMM shapes and throughput.

The project provides:

- a masked Triton single-GEMM kernel;
- a one-launch persistent grouped-GEMM kernel over distinct expert pointers;
- equal-FLOP EP/TP workload construction for packed W1/W3 Gate+Up and W2 Down projections;
- PyTorch `torch.mm` single-GEMM baselines and a native `torch.nn.functional.grouped_mm` MoE baseline;
- hot and rotating-cold workspace regimes;
- CSV provenance, plots, summaries, and offline result validation.

> This is not an end-to-end MoE benchmark. Routing, token permutation, load imbalance, shared experts, collectives, and network communication are excluded.

## Equal-FLOP comparison

The workload starts after a *logical* top-k expansion. The notation is:

```text
T   = original tokens before routing
X   = T * topk expert-assignment rows after top-k expansion
E   = num_experts
P   = parallel_size
M_e = X / E = T * topk / E rows per expert under uniform routing
```

If an input is described as `[x,H]` in this benchmark, `x` means `X`, the number of top-k-expanded expert-assignment rows, not the original token count `T`. Consequently, one TP rank processes `X` rows distributed uniformly across all `E` experts, while one EP rank processes `X/P` rows distributed uniformly across its `E/P` local experts. Both cases therefore use the same `M_e=X/E` rows per expert:

```text
TP rank: E   expert inputs * M_e rows = X rows
EP rank: E/P expert inputs * M_e rows = X/P rows
```

There is currently no router implementation. The benchmark does not compute routing scores, select experts, construct routing indices, or perform token permutation. Instead, it assumes perfectly uniform routing and directly allocates already partitioned `[M_e,K]` input matrices for each expert. These separate matrices are the compute-only equivalent of slicing an expert-sorted aggregate input; routing imbalance and the cost of producing that layout are outside the timed region and outside the current scope.

`--tokens` specifies the original pre-routing count `T`; the benchmark derives `X=T*topk` and `M_e=X/E`. `--moe-ms` specifies `M_e` directly and derives an integral `T=M_e*E/topk` for provenance.

For the fused W1/W3 Gate+Up projection, `F_local` is `F` under EP and `F/P`
under TP. Gate and Up weights are packed along the output dimension in vLLM's
W13 order, so one grouped Triton launch produces `[gate_local, up_local]`:

```text
EP rank: num_experts/P GEMMs of [M_e,H] @ [H,2F]
TP rank: num_experts   GEMMs of [M_e,H] @ [H,2F/P]
```

For W2 Down:

```text
EP rank: num_experts/P GEMMs of [M_e,F]   @ [F,H]
TP rank: num_experts   GEMMs of [M_e,F/P] @ [F/P,H]
```

The Gate+Up timing covers the fused packed GEMM only. The following SiLU and
elementwise `SiLU(gate) * up` are deliberately outside the timed region, as
are routing and communication. The MoE experiment rejects non-divisible
configurations and verifies exact per-rank FLOP equality before allocating each
case's GPU workspaces. W1/W3 therefore records twice the useful GEMM FLOPs of
W2 for matching `M_e`, `H`, and `F`.

## Requirements

- Linux
- Python 3.10–3.12
- an NVIDIA GPU and driver supported by the selected PyTorch build
- PyTorch 2.13.x and Triton `>=3.7.1,<3.8`

Create and populate an environment with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

For tests and linting:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

The default PyTorch 2.13 Linux wheel uses CUDA 13.0. Choose a different official wheel only when required by the installed driver and GPU; do not copy a wheel selection from another machine without checking compatibility.

## Quick start

```bash
python benchmark.py --experiment shapes
python benchmark.py --experiment moe
python benchmark.py --experiment heatmap
python benchmark.py                         # all experiments
```

Useful examples:

```bash
# Use a logical device after CUDA_VISIBLE_DEVICES remapping
python benchmark.py --device cuda:1 --experiment shapes

# FP32 comparison with TF32 enabled consistently for Triton and PyTorch
python benchmark.py --dtype fp32 --input-precision tf32 --experiment shapes

# Override MoE dimensions
python benchmark.py --experiment moe --parallel-size 8 \
  --num-experts 256 --topk 8 --hidden-size 7168 --ffn-size 2048

# One tokens-per-expert point derived from the token count
python benchmark.py --experiment moe --tokens 4096

# Rotating resident workspaces
python benchmark.py --cache-mode cold --cold-buffers 2

# Skip plotting dependencies/work
python benchmark.py --experiment shapes --no-plots
```

`--device` accepts logical CUDA names such as `cuda` or `cuda:1`. `CUDA_VISIBLE_DEVICES` remains authoritative. Hardware provenance is collected from the selected logical PyTorch device rather than assuming physical GPU 0.

Run `python benchmark.py --help` for all shape, timing, memory, baseline, and output controls.

## Model configurations

The default is `model_configs/deepseek-v3.json`. The loader also supports the common schemas represented by every JSON file under `model_configs/`, including top-level and multimodal `text_config` layouts. It recognizes these field families:

- expert width: `moe_intermediate_size` or `intermediate_size`;
- expert count: `n_routed_experts`, `num_local_experts`, or `num_experts`;
- routed top-k: `num_experts_per_tok`, `num_experts_per_token`, or `router_top_k`.

Use another bundled or external configuration with:

```bash
python benchmark.py --model-config model_configs/mixtral-8x7b.json --experiment moe
```

External absolute paths are accepted, but persisted provenance stores only a safe display name instead of the local directory hierarchy. Upstream sources and revisions are documented in [`model_configs/README.md`](model_configs/README.md).

## Kernels and timing

### Scheduling modes

- `grouped`: the MoE Triton provider launches one bounded persistent CTA grid for all local expert GEMMs. W1/W3 packs Gate and Up into a single `2F_local` output and therefore remains one Triton launch, matching vLLM's W13 layout. Equal contiguous shapes use an optimized constexpr fast path; heterogeneous shapes use device-side dimensions and strides. A one-problem group dispatches to the standard Triton matmul kernel instead of retaining grouped-scheduler overhead.
- `torch`: one native `torch.nn.functional.grouped_mm` call over a 2D expert-sorted activation matrix, 3D expert weights, and cumulative `int32` offsets. It is enabled by default and removable with `--no-torch-baseline`; there is no sequential per-expert fallback. The public API can silently fall back to serial per-expert `mm` calls, so API availability alone is insufficient. With PyTorch 2.13, this benchmark only admits BF16 on the SM90/SM100 architecture families and fewer than 1024 groups, then profiles the exact workload outside timing to require one CUTLASS grouped compute kernel and no sequential matmul or device transfer. Missing CUPTI traces or unrecognized execution paths produce `status=skipped`. FP16 (the default) and FP32 torch MoE baselines are explicitly skipped; use `--dtype bf16` on a supported device for a native grouped baseline. Dtypes are never silently converted. Backing storage is row-padded when necessary to satisfy the operator's 16-byte stride alignment without changing logical GEMM dimensions or useful FLOPs.

The standalone shape and heatmap experiments still use the single-GEMM Triton kernel; it is not emitted as a MoE scheduling curve.

Grouped Triton autotuning uses up to 12 base configurations (previously 8) and at most 36 configurations including distinct persistent grid sizes. A greedy coverage pass preserves M/N/K tile sizes, 2/4/8 warps, 2–5 pipeline stages, and 1/4/8 M-group traversal choices before analytical-rank filling. The portable tile families in Triton tutorial 08 are represented in the pool. Standalone GEMM remains bounded to 10 candidates. It searches `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, warp/stage counts, and persistent CTA count while accounting for problem count, total output tiles, padding, and the active device's SM count. Hot-mode candidates are measured by Triton's autotuner with a custom CUDA Graph timer that preserves the hot cache regime (up to 9 samples, 25 ms target per candidate instead of the default cache-flushing timer); cold-mode candidates are measured while rotating across the complete resident workspace ring. Compilation, autotuning, allocation, random initialization, reference calculation, and correctness checks occur before timing. Timed repetitions are captured into a CUDA Graph, with explicit CUDA event nodes around each GEMM call inside the graph. Samples come from one graph replay: Python dispatch and CPU submission gaps cannot occur between a sample's start and end. Capture, graph initialization, allocator warmup, and graph warmup are untimed. Capture failures are reported rather than falling back to eager timing. Rotating-cold candidate selection uses the same graph timing method.

`--warmup` (default 25) and `--repeat` (default 100) are requested counts. After an untimed pilot, warmups are capped to approximately 250 ms with a minimum of 2, and repetitions are capped toward `--target-timing-ms` (default 2000 ms) with a minimum of 3. The timing target is not a strict wall-clock limit. Counts are rounded up to complete workspace rings so every cold buffer is measured, even when the requested repeat count is smaller than the ring. CSV rows store requested and actual counts; actual counts exclude compilation, the pilot, and one final graph warmup. Schema 1.3 records `timing_method=cuda_graph_events`; parallel sweeps reject mixing this method with old eager-event results.

The PyTorch public grouped API returns a library-managed output and has no `out=` parameter. Before each repeated call the prior result is released, so after the untimed warmup PyTorch's caching allocator reuses the same output storage. Graph capture performs host allocator bookkeeping before timing; graph replay uses the captured allocator storage. CUDA event nodes measure stream execution within the replay. CSV rows record `output_preallocated=false` for this provider and `true` for the Triton provider.

FP32 precision is explicit:

- `--input-precision ieee` disables TF32 for the PyTorch baseline and requests IEEE input precision from Triton;
- `--input-precision tf32` requires `--dtype fp32` and enables the matching PyTorch policy.

Every resident workspace is correctness-checked before timing. After autotuning, Triton outputs are filled with NaNs and the selected configuration is launched alone before reference comparison, preventing stale candidate output from hiding omitted writes. Homogeneous kernels keep consecutive indices and mask M/N/K tails; they do not assert false contiguity for wrapped edge indices.

### Cache regimes

- `hot`: repeatedly uses one resident A/B/C workspace.
- `cold`: rotates complete resident workspaces and metadata. Triton outputs are preallocated; the PyTorch provider reuses warmed caching-allocator output storage because its public API has no `out=` parameter.

Rotating buffers reduce cache reuse but do not prove L2 misses; hardware counters are required for that claim.

### Memory admission

Each case estimates one complete workspace against current free memory and `--memory-fraction`. The allocator cache is flushed before admission. OOM retries reduce only the rotating workspace count, never the requested GEMM shape. Cases that cannot fit one hot or two cold workspaces are written as `status=skipped`.

## Outputs

Default outputs are generated under:

```text
results/environment.json
results/gemm_shape_sweep.csv
results/moe_w1.csv
results/moe_w2.csv
results/gemm_heatmap.csv
results/summary.md
results/validation.json
plots/figure_*.png
```

These files contain machine-specific hardware, software, timing, and run metadata. They are recursively ignored by Git; only `results/.gitkeep` and `plots/.gitkeep` are versioned. Review generated data before publishing it elsewhere.

Replot or revalidate persisted output without running kernels:

```bash
python plot_results.py --results-dir results --plots-dir plots
python validate_results.py --results-dir results --plots-dir plots
```

Validation recomputes derivable FLOPs and TFLOPS, checks the packed Gate+Up and
Down projection contracts, verifies correctness and metrics for `status=ok`
rows, treats `status=invalid` and `status=error` rows as failures, counts
`status=skipped` rows, and verifies matched MoE FLOPs. Every successful Triton
MoE row must record a selected workload-aware grouped configuration, and every
successful PyTorch MoE row must record verified grouped execution. Schema 1.4 separates `api_calls_per_iteration=1` from the profiler-observed `launches_per_iteration` and `grouped_compute_launches=1`; the library may also launch metadata preparation kernels, which remain part of its timing. Validation
rejects the removed MoE `single` and sequential torch modes and checks generated
PNG signatures/counts.

## Parallel-size sweeps

Write each run to a separate directory:

```bash
for p in 2 4 8 16 32; do
  python benchmark.py --experiment moe --parallel-size "$p" \
    --results-dir "results/parallel_sweep/p$p" \
    --plots-dir "plots/parallel_sweep/p$p"
done
```

Aggregation requires explicit source mappings; it never assumes local baseline directories:

```bash
python plot_parallel_sweep.py \
  --run 2=results/parallel_sweep/p2 \
  --run 4=results/parallel_sweep/p4 \
  --run 8=results/parallel_sweep/p8 \
  --run 16=results/parallel_sweep/p16 \
  --run 32=results/parallel_sweep/p32
```

The aggregator verifies common model, dtype, cache, timing, software, and
hardware controls before generating its CSV, summary, validation report, and
six scaling plots. The summary includes exact TP Gate+Up, Down, and combined
GEMM latency for every matched `(P, M_e)` point. Labels and summaries are
derived from source provenance rather than hard-coded to a model or GPU.

## Metrics

```text
FLOPs = 2*M*K*N
ideal bytes = sizeof(dtype) * (M*K + K*N + M*N)
arithmetic intensity = FLOPs / ideal bytes
TFLOPS = total useful FLOPs / latency
```

The byte model assumes one A read, one B read, and one C write. It is not measured HBM traffic. Tile counts, waves, and padded-tile efficiencies are explanatory scheduling indicators, not an occupancy or cache simulator. `--peak-tflops` is never guessed; provide a positive value only when you want normalized theoretical-peak reporting.

## Testing

Hardware-independent checks:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest -q tests/test_cpu.py
python -m ruff check .
```

CUDA/Triton checks:

```bash
python -m pytest -q tests/test_gpu.py
python benchmark.py --experiment moe --moe-ms 16,32 --repeat 5 --warmup 2 --no-plots
```

The representative MoE run is also the minimum performance sanity check for grouped tile selection and autotuning; inspect the selected configuration, scheduler, latency, and the optional native PyTorch grouped baseline in its CSV output. GPU tests are skipped when CUDA PyTorch is unavailable. See [`AGENTS.md`](AGENTS.md) for contributor and automated-agent guidance.

## Official Grouped GEMM comparison

Run the independent comparison against the pinned, unmodified portable kernel
from [Triton tutorial 08, v3.7.1](https://github.com/triton-lang/triton/blob/v3.7.1/python/tutorials/08-grouped-gemm.py):

```bash
python compare_grouped_gemm.py
python compare_grouped_gemm.py --cache-mode cold --cold-buffers 2
python compare_grouped_gemm.py --shape 128,7168,4096 --experts 32
python compare_grouped_gemm.py --moe-config model_configs/deepseek-v3.json \
  --parallel-size 8 --moe-ms 128,256
```

Defaults cover homogeneous, heterogeneous, and multi-round persistent workloads.
Both implementations use exactly the same resident A/B/C buffers, useful FLOPs,
FP16 inputs, and CUDA Graph timing. Every measured candidate is checked against
an FP32 reference after poisoning outputs. Both searches remeasure at most three
finalists, and final measurements reverse provider order in a second round.
The tutorial's four portable tile families use device-relative grids instead of
its fixed machine-specific CTA counts. Its kernel has no masks, so unsupported
tail shapes are explicitly skipped without changing logical dimensions. This
comparison does not include the tutorial's separate TMA kernel.

The JSON report defaults to `results/grouped_tutorial_comparison.json`. It records
source hashes, hardware/software, selected configurations, candidate counts,
correctness errors, timings, and `project_over_tutorial_latency` (lower is better).
Use `--max-slowdown 1.10` to fail if the project exceeds tutorial latency by more
than 10% on a measured point. No performance threshold is assumed by default.
Exit codes are 0 for measured success, 2 for a correctness/runtime/threshold
failure, and 3 when every case is skipped, including when CUDA is unavailable.
Skipped cases are never evidence of correctness or speed; inspect their reasons.
An AST checksum CPU test guards the copied tutorial kernel, and GPU tests compare
both kernels with matching configurations on homogeneous and heterogeneous inputs.

## Limitations

- synthetic dense FP16/BF16/FP32 data only;
- no deployed FP8/quantized kernel emulation;
- no router, communication, or end-to-end layer timing;
- grouped heterogeneous scheduling still scans problem metadata in each persistent CTA;
- no SM90-specific TMA/warp-specialized implementation;
- results depend on clocks, thermals, competing processes, software versions, autotuning, and cache state.

Candidate limits bound measurement and search work, not JIT compilation time. The selected configuration is best among the measured candidates; this is not a guarantee of global optimality. Do not reuse throughput numbers or selected Triton configurations across machines. Regenerate results on the target environment.

## License

Project code is available under the [MIT License](LICENSE). Bundled model configuration data remains subject to upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
