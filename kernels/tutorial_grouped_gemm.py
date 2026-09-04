"""Pinned portable kernel from Triton tutorial 08, with a benchmark adapter.

The kernel body is unchanged from v3.7.1. Only the wrapper, candidate selection
and device-relative grid configuration are local. No TMA tutorial code is used.
"""

# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    torch = None
    triton = None
    tl = None

from .grouped_gemm import GroupedWorkspace
from .matmul import KernelConfig

TUTORIAL_SOURCE = "https://github.com/triton-lang/triton/blob/v3.7.1/python/tutorials/08-grouped-gemm.py"
TUTORIAL_SOURCE_SHA256 = "16647534382b116fda2a034acef5fb59bba8d803df56863596f4bc7b444abb16"
TUTORIAL_KERNEL_AST_SHA256 = "28010a2f5e3f5ad060f7df1391586217151086d125653606cbd49df697c45392"

grouped_matmul_kernel = None
if triton is not None:
    @triton.jit
    def grouped_matmul_kernel(
        # device tensor of matrices pointers
        group_a_ptrs,
        group_b_ptrs,
        group_c_ptrs,
        # device tensor of gemm sizes. its shape is [group_size, 3]
        # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
        group_gemm_sizes,
        # device tensor of leading dimension sizes. its shape is [group_size, 3]
        # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
        g_lds,
        # number of gemms
        group_size,
        # number of virtual SM
        NUM_SM: tl.constexpr,
        # tile sizes
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        tile_idx = tl.program_id(0)
        last_problem_end = 0
        for g in range(group_size):
            # get the gemm size of the current problem
            gm = tl.load(group_gemm_sizes + g * 3)
            gn = tl.load(group_gemm_sizes + g * 3 + 1)
            gk = tl.load(group_gemm_sizes + g * 3 + 2)
            num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles
            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                # pick up a tile from the current gemm problem
                k = gk
                lda = tl.load(g_lds + g * 3)
                ldb = tl.load(g_lds + g * 3 + 1)
                ldc = tl.load(g_lds + g * 3 + 2)
                a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
                b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
                c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):  # noqa: PIE808 - preserve upstream AST
                    # hint to Triton compiler to do proper loop pipelining
                    tl.multiple_of(a_ptrs, [16, 16])
                    tl.multiple_of(b_ptrs, [16, 16])
                    # assume full tile for now
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K * ldb
                c = accumulator.to(tl.float16)

                offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

                # assumes full tile for now
                tl.store(c_ptrs, c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

            # get ready to go to the next gemm problem
            last_problem_end = last_problem_end + num_tiles


@dataclass
class TutorialWorkspace:
    grouped: GroupedWorkspace
    leading_dimensions: object


def tutorial_candidate_configs(shapes, sm_count: int) -> tuple[KernelConfig, ...]:
    """Upstream portable tiles; replace fixed device sizes by relative grids.

    The upstream kernel has no edge masks. Filter incompatible tiles without
    changing a logical dimension, a pointer or the useful FLOP count.
    """
    if not shapes or sm_count <= 0 or any(min(shape) <= 0 for shape in shapes):
        raise ValueError("positive tutorial shapes and SM count required")
    candidates = []
    for bm, bn, bk in ((128, 128, 32), (64, 64, 32), (128, 128, 64), (64, 128, 64)):
        if any(m % bm or n % bn or k % bk for m, k, n in shapes):
            continue
        tiles = sum((m // bm) * (n // bn) for m, _, n in shapes)
        for multiplier in ((1, 2) if tiles > sm_count else (1,)):
            candidates.append(KernelConfig(bm, bn, bk, 1, 4, 3, multiplier))
    return tuple(candidates)


def prepare_tutorial_workspace(workspace: GroupedWorkspace) -> TutorialWorkspace:
    if torch is None or any(t.dtype != torch.float16 or not t.is_contiguous()
                            for t in (*workspace.a, *workspace.b, *workspace.c)):
        raise ValueError("unmodified tutorial kernel requires contiguous FP16 matrices")
    if not tutorial_candidate_configs(workspace.shapes, workspace.sm_count):
        raise ValueError("tutorial kernel requires full tiles; logical shapes will not be padded")
    leading = [[a.stride(0), b.stride(0), c.stride(0)] for a, b, c in workspace.problem_tensors()]
    if any(value >= 2**31 for strides in leading for value in strides):
        raise ValueError("tutorial leading dimensions exceed int32")
    return TutorialWorkspace(workspace, torch.tensor(leading, dtype=torch.int32, device=workspace.a[0].device))


def launch_tutorial(workspace: TutorialWorkspace, config: KernelConfig) -> None:
    grouped = workspace.grouped
    if config not in tutorial_candidate_configs(grouped.shapes, grouped.sm_count):
        raise ValueError("unsafe or unsupported tutorial configuration")
    grid_size = grouped.sm_count * config.cta_multiplier
    grouped_matmul_kernel[(grid_size,)](
        grouped.a_ptrs, grouped.b_ptrs, grouped.c_ptrs, grouped.dims,
        workspace.leading_dimensions, grouped.problem_count,
        NUM_SM=grid_size, BLOCK_SIZE_M=config.block_m, BLOCK_SIZE_N=config.block_n,
        BLOCK_SIZE_K=config.block_k, num_warps=config.num_warps, num_stages=config.num_stages,
    )
