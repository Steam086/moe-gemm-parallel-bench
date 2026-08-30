# AGENTS.md

## Scope

This repository benchmarks compute-only MoE GEMM shapes with PyTorch and Triton. Preserve the distinction between useful FLOPs and end-to-end MoE performance: routing, token movement, collectives, and networking are out of scope unless a change explicitly adds and documents them.

## Code map

- `benchmark.py`: CLI and run provenance.
- `benchmark_shapes.py`: shape and heatmap experiments.
- `benchmark_moe.py`: equal-FLOP EP/TP experiments.
- `kernels/`: Triton single and persistent grouped GEMM kernels.
- `utils/`: timing, hardware, config, I/O, and metrics helpers.
- `plot_*.py`, `validate_results.py`: offline reporting and validation.
- `tests/test_cpu.py`: hardware-independent contracts.
- `tests/test_gpu.py`: CUDA/Triton correctness checks.

## Required practices

- Do not hard-code usernames, absolute checkout paths, hostnames, GPU models, SM counts, memory sizes, CUDA device 0, or local result directories.
- Treat CUDA indices as logical indices after `CUDA_VISIBLE_DEVICES` remapping.
- Keep allocation, random initialization, compilation, autotuning, references, and correctness checks outside timed regions.
- Keep EP/TP useful-FLOP equality checks exact and use Python integers.
- Match PyTorch and Triton FP32/TF32 policies when comparing providers.
- Do not weaken correctness masks or dtype-aware tolerances to improve reported speed.
- Generated `results/` and `plots/` content is machine-specific and must remain untracked. Keep only each directory's `.gitkeep`.
- Do not publish environment/result artifacts without checking them for paths and machine metadata.
- Preserve clear `status=skipped` rows rather than silently changing requested shapes after OOM.

## Validation

Run from the repository root:

```bash
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest -q tests/test_cpu.py        # when dev dependencies are installed
python -m ruff check .                       # when dev dependencies are installed
```

On a CUDA/Triton host also run:

```bash
python -m pytest -q tests/test_gpu.py
python benchmark.py --experiment shapes --shape-ms 16,32 --repeat 5 --warmup 2 --no-plots
python benchmark.py --experiment moe --moe-ms 16,32 --repeat 5 --warmup 2 --no-plots
```

A full default benchmark is expensive and is not required for documentation-only changes. If GPU validation is unavailable, report that explicitly; never imply it passed.

## Documentation and commits

Update `README.md` when CLI flags, output schemas, timing semantics, model-config support, or reproducibility requirements change. Keep commits focused and report commands run, skipped checks, and residual hardware-specific risks.
