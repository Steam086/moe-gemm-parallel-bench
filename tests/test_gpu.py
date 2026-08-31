from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from kernels.grouped_gemm import build_fused_gate_up_workspace, build_grouped_workspace, launch_grouped
from kernels.matmul import last_matmul_config, launch_matmul, select_config
from kernels.torch_grouped_gemm import build_torch_grouped_workspace, launch_torch_grouped
from utils.benchmark import assert_close


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA PyTorch unavailable")
class TritonCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_single_odd_shapes_all_dtypes(self) -> None:
        dtypes = (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32))
        for dtype_name, dtype in dtypes:
            if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            for m, k, n in ((1, 5, 3), (17, 13, 31), (33, 70, 65)):
                a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.1
                b = torch.randn((k, n), device="cuda", dtype=dtype) * 0.1
                c = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
                fallback = select_config(m, n, k)
                launch_matmul(a, b, c, fallback)
                torch.cuda.synchronize()
                selected = last_matmul_config(fallback)
                self.assertIn(selected.block_m, (16, 32, 64, 128, 256))
                self.assertIn(selected.block_n, (32, 64, 128, 256))
                self.assertIn(selected.block_k, (32, 64, 128))
                assert_close(c, a @ b, dtype_name)
                self.assertTrue(torch.isfinite(c).all())

    def test_single_strided_fp32_tf32(self) -> None:
        m, k, n = 17, 70, 65
        a = (torch.randn((m, k * 2), device="cuda", dtype=torch.float32) * 0.1)[:, ::2]
        b = (torch.randn((k, n * 2), device="cuda", dtype=torch.float32) * 0.1)[:, ::2]
        storage = torch.full((m, n * 2), float("nan"), device="cuda", dtype=torch.float32)
        c = storage[:, ::2]
        launch_matmul(a, b, c, input_precision="tf32")
        torch.cuda.synchronize()
        assert_close(c, a @ b, "fp32")
        self.assertTrue(torch.isfinite(c).all())

    def test_grouped_distinct_pointers_shapes_and_dtypes(self) -> None:
        shapes = [(1, 5, 3), (17, 13, 31), (9, 70, 65)]
        config = select_config(17, 65, 70)
        dtypes = (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32))
        for dtype_name, dtype in dtypes:
            if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            workspace = build_grouped_workspace(shapes, dtype, seed=99, config=config)
            self.assertEqual(len({x.data_ptr() for x in workspace.a}), len(shapes))
            self.assertEqual(len({x.data_ptr() for x in workspace.b}), len(shapes))
            launch_grouped(workspace)
            torch.cuda.synchronize()
            self.assertEqual(workspace.scheduler, "generic_persistent")
            self.assertIn(workspace.config.block_k, (32, 64, 128))
            self.assertIn(workspace.config.num_warps, (2, 4, 8))
            for a, b, c in zip(workspace.a, workspace.b, workspace.c):
                assert_close(c, a @ b, dtype_name)

    def test_grouped_homogeneous_persistent_fast_path(self) -> None:
        shapes = [(17, 13, 31)] * 3
        workspace = build_grouped_workspace(shapes, torch.float16, seed=101)
        self.assertTrue(workspace.homogeneous)
        launch_grouped(workspace)
        torch.cuda.synchronize()
        self.assertEqual(workspace.scheduler, "homogeneous_persistent")
        for a, b, c in zip(workspace.a, workspace.b, workspace.c):
            assert_close(c, a @ b, "fp16")

    def test_fused_gate_up_packed_output_in_one_grouped_launch(self) -> None:
        local_ffn = 31
        workspace = build_fused_gate_up_workspace([(17, 13, 2 * local_ffn)] * 3, torch.float16, seed=102)
        self.assertEqual(workspace.operation, "gate_up")
        launch_grouped(workspace)
        torch.cuda.synchronize()
        self.assertEqual(workspace.scheduler, "homogeneous_persistent")
        for a, packed_weight, packed_output in zip(workspace.a, workspace.b, workspace.c):
            reference = torch.cat(
                (a @ packed_weight[:, :local_ffn], a @ packed_weight[:, local_ffn:]),
                dim=1,
            )
            assert_close(packed_output, reference, "fp16")

    def test_one_problem_group_dispatches_standard_matmul(self) -> None:
        workspace = build_grouped_workspace([(17, 70, 65)], torch.float16, seed=103)
        launch_grouped(workspace)
        torch.cuda.synchronize()
        self.assertEqual(workspace.scheduler, "single_problem_matmul")
        assert_close(workspace.c[0], workspace.a[0] @ workspace.b[0], "fp16")

    @unittest.skipUnless(
        torch is not None and hasattr(torch.nn.functional, "grouped_mm"),
        "public torch grouped_mm unavailable",
    )
    def test_torch_native_grouped_mm_workspace(self) -> None:
        if torch.cuda.get_device_capability() < (8, 0) or not torch.cuda.is_bf16_supported():
            self.skipTest("torch grouped_mm BF16 requires SM80+")
        shapes = [(17, 13, 31)] * 3
        workspace = build_torch_grouped_workspace(shapes, torch.bfloat16, seed=104)
        self.assertEqual(workspace.offs.tolist(), [17, 34, 51])
        launch_torch_grouped(workspace)
        torch.cuda.synchronize()
        self.assertEqual(workspace.scheduler, "torch_grouped_mm")
        for a, b, c in workspace.problem_tensors():
            assert_close(c, a @ b, "bf16")


if __name__ == "__main__":
    unittest.main()
