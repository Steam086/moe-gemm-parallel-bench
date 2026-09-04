# Third-party notices

The JSON files in `model_configs/` are derived from model configuration files published in the Hugging Face repositories and revisions listed in [`model_configs/README.md`](model_configs/README.md). Model configuration data and model names remain subject to the notices, licenses, and acceptable-use terms of their respective upstream projects. The repository's MIT license does not replace those upstream terms.

`kernels/tutorial_grouped_gemm.py` contains the portable `grouped_matmul_kernel`
from [Triton tutorial 08, v3.7.1](https://github.com/triton-lang/triton/blob/v3.7.1/python/tutorials/08-grouped-gemm.py),
copyright 2023–2025 NVIDIA Corporation & Affiliates, under the MIT license. The
copyright and permission notice are retained in that file. Its kernel AST is
unchanged; the local adapter replaces machine-specific grid candidates with
device-relative grids, filters unsafe partial tiles, and preallocates metadata.
The TMA tutorial kernel and upstream benchmark harness are not included. The
full upstream source SHA256 and kernel AST SHA256 are recorded in the adapter.

PyTorch, Triton, NumPy, pandas, Matplotlib, and pytest are otherwise separate dependencies distributed under their own licenses.
