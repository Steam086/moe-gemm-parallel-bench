# MoE GEMM Parallel Benchmark

A compute-only CUDA/Triton benchmark for studying how MoE Tensor Parallelism (TP) and Expert Parallelism (EP) can have equal useful FLOPs but different GEMM shapes and throughput.

The project provides:

- a masked Triton single-GEMM kernel;
- a one-launch persistent grouped-GEMM kernel over distinct expert pointers;
- equal-FLOP EP/TP workload construction for W1 and W2 projections;
- PyTorch `torch.mm` baselines under the same FP32/TF32 policy;
- hot and rotating-cold workspace regimes;
- CSV provenance, plots, summaries, and offline result validation.

> This is not an end-to-end MoE benchmark. Routing, token permutation, load imbalance, shared experts, collectives, and network communication are excluded.

## Equal-FLOP comparison

For uniform routing, each expert receives:

```text
M_e = tokens * topk / num_experts
```

For W1:

```text
EP rank: num_experts/P GEMMs of [M_e,H] @ [H,F]
TP rank: num_experts   GEMMs of [M_e,H] @ [H,F/P]
```

For W2:

```text
EP rank: num_experts/P GEMMs of [M_e,F]   @ [F,H]
TP rank: num_experts   GEMMs of [M_e,F/P] @ [F/P,H]
```

The benchmark rejects non-divisible configurations and verifies exact per-rank FLOP equality before allocating GPU memory.

## Requirements

- Linux
- Python 3.10–3.12
- an NVIDIA GPU and driver supported by the selected PyTorch build
- PyTorch 2.6–2.7 and Triton 3.2–3.3

Install a CUDA-compatible PyTorch wheel first. For example:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements.txt
```

For tests and linting:

```bash
pip install -r requirements-dev.txt
```

The CUDA wheel/index above is only an example. Choose versions compatible with your driver and GPU. Do not copy a wheel selection from another machine without checking compatibility.

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

- `grouped`: the MoE Triton provider launches one bounded persistent CTA grid for all local expert GEMMs. Equal contiguous shapes use an optimized constexpr fast path; heterogeneous shapes use device-side dimensions and strides. A one-problem group dispatches to the standard Triton matmul kernel instead of retaining grouped-scheduler overhead.
- `torch`: sequential `torch.mm` baseline, enabled by default and removable with `--no-torch-baseline`.

The standalone shape and heatmap experiments still use the single-GEMM Triton kernel; it is not emitted as a MoE scheduling curve.

Grouped Triton autotuning uses a bounded workload-aware candidate set. It searches `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, warp/stage counts, and persistent CTA count while accounting for problem count, total output tiles, padding, and the active device's SM count. Hot-mode candidates are measured by Triton's autotuner; cold-mode candidates are measured while rotating across the complete resident workspace ring. Compilation, autotuning, allocation, random initialization, reference calculation, and correctness checks occur before timing. Timed repetitions are enqueued back-to-back with CUDA events and synchronized once at the end, avoiding a host synchronization between every sample.

FP32 precision is explicit:

- `--input-precision ieee` disables TF32 for the PyTorch baseline and requests IEEE input precision from Triton;
- `--input-precision tf32` requires `--dtype fp32` and enables the matching PyTorch policy.

Every resident rotating-cold workspace is correctness-checked before timing.

### Cache regimes

- `hot`: repeatedly uses one resident A/B/C workspace.
- `cold`: rotates complete preallocated workspaces and pointer tables without allocation or copies in the timed region.

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

Validation recomputes derivable FLOPs and TFLOPS, checks correctness/status fields, verifies matched MoE FLOPs, requires every successful Triton MoE row to record a selected workload-aware grouped configuration and supported scheduler, rejects the removed MoE `single` mode, and checks generated PNG signatures/counts.

## Parallel-size sweeps

Write each run to a separate directory:

```bash
for p in 2 4 8 16; do
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
  --run 16=results/parallel_sweep/p16
```

The aggregator verifies common model, dtype, cache, timing, software, and hardware controls before generating its CSV, summary, validation report, and six scaling plots. Labels and summaries are derived from source provenance rather than hard-coded to a model or GPU.

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

The representative MoE run is also the minimum performance sanity check for grouped tile selection and autotuning; inspect the selected configuration, scheduler, latency, and the optional `torch` baseline in its CSV output. GPU tests are skipped when CUDA PyTorch is unavailable. See [`AGENTS.md`](AGENTS.md) for contributor and automated-agent guidance.

## Limitations

- synthetic dense FP16/BF16/FP32 data only;
- no deployed FP8/quantized kernel emulation;
- no router, communication, or end-to-end layer timing;
- grouped heterogeneous scheduling still scans problem metadata in each persistent CTA;
- no SM90-specific TMA/warp-specialized implementation;
- results depend on clocks, thermals, competing processes, software versions, autotuning, and cache state.

Do not reuse throughput numbers or selected Triton configurations across machines. Regenerate results on the target environment.

## License

Project code is available under the [MIT License](LICENSE). Bundled model configuration data remains subject to upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
