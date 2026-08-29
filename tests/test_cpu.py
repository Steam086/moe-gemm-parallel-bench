from __future__ import annotations

import argparse
import unittest

from benchmark import main, parser
from kernels.grouped_gemm import grouped_candidate_configs
from kernels.matmul import matmul_candidate_configs
from plot_parallel_sweep import parse_run_spec
from utils.config import load_model_defaults
from utils.io import CSV_FIELDS
from utils.metrics import arithmetic_intensity, gemm_flops, moe_shapes, tile_metrics, verify_ep_tp_flops
from validate_results import _derivable_metrics_valid


class CpuMathTests(unittest.TestCase):
    def test_deepseek_v3_mapping(self) -> None:
        config = load_model_defaults("model_configs/deepseek-v3.json")
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual(config.ffn_size, 2048)  # routed expert, not dense intermediate_size
        self.assertEqual(config.num_experts, 256)
        self.assertEqual(config.topk, 8)

    def test_all_bundled_model_config_schemas(self) -> None:
        expected = {
            "deepseek-v3.json": (7168, 2048, 256, 8),
            "deepseek-v4-flash.json": (4096, 2048, 256, 6),
            "glm-5.2.json": (6144, 2048, 256, 8),
            "gpt-oss-120b.json": (2880, 2880, 128, 4),
            "kimi-k3.json": (7168, 3072, 896, 16),
            "llama-4-maverick-17b-128e.json": (5120, 8192, 128, 1),
            "llama-4-scout-17b-16e.json": (5120, 8192, 16, 1),
            "mixtral-8x7b.json": (4096, 14336, 8, 2),
            "qwen3-235b-a22b.json": (4096, 1536, 128, 8),
            "qwen3-30b-a3b.json": (2048, 768, 128, 8),
        }
        for filename, dimensions in expected.items():
            with self.subTest(filename=filename):
                config = load_model_defaults(f"model_configs/{filename}")
                self.assertEqual(
                    (config.hidden_size, config.ffn_size, config.num_experts, config.topk),
                    dimensions,
                )

    def test_ep_tp_flops_equal_w1_and_w2(self) -> None:
        for projection in ("W1", "W2"):
            total = verify_ep_tp_flops(projection, 17, 7168, 2048, 256, 8)
            ep = sum(gemm_flops(*x) for x in moe_shapes(projection, "EP", 17, 7168, 2048, 256, 8))
            tp = sum(gemm_flops(*x) for x in moe_shapes(projection, "TP", 17, 7168, 2048, 256, 8))
            self.assertEqual(total, ep)
            self.assertEqual(ep, tp)

    def test_shape_and_tile_metrics(self) -> None:
        self.assertEqual(gemm_flops(2, 3, 4), 48)
        self.assertGreater(arithmetic_intensity(16, 32, 64, 2), 0)
        metrics = tile_metrics([(17, 31, 65)] * 3, 16, 64, 32, 30)
        self.assertEqual(metrics["total_output_tiles_per_rank"], 12)
        self.assertEqual(metrics["estimated_waves"], 1)

    def test_divisibility_rejected(self) -> None:
        with self.assertRaises(ValueError):
            moe_shapes("W1", "TP", 1, 8, 10, 8, 4)

    def test_shape_family_autotune_coverage_is_bounded(self) -> None:
        tiny = matmul_candidate_configs(8, 32, 4096)
        wide = matmul_candidate_configs(512, 8192, 7168)
        skinny_n = matmul_candidate_configs(512, 64, 7168)
        self.assertLessEqual(len(tiny), 10)
        self.assertLessEqual(len(wide), 10)
        self.assertTrue(any(cfg.num_warps == 2 for cfg in tiny))
        self.assertTrue(any(cfg.block_n == 256 for cfg in wide))
        self.assertTrue(any(cfg.block_k == 128 for cfg in wide))
        self.assertTrue(any(cfg.block_m == 256 for cfg in skinny_n))
        grouped = grouped_candidate_configs(16, 256, 7168)
        self.assertEqual({cfg.cta_multiplier for cfg in grouped}, {1, 2, 4})

    def test_baseline_and_provenance_defaults(self) -> None:
        self.assertTrue(parser().parse_args([]).torch_baseline)
        self.assertFalse(parser().parse_args(["--no-torch-baseline"]).torch_baseline)
        for field in (
            "scheduler",
            "cta_multiplier",
            "best_measured_ratio",
            "near_best",
            "autotune_candidates_tested",
        ):
            self.assertIn(field, CSV_FIELDS)

    def test_non_positive_cli_overrides_are_rejected(self) -> None:
        for option in ("--hidden-size", "--ffn-size", "--num-experts", "--topk", "--heatmap-k"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                main([option, "0", "--no-plots"])
        with self.assertRaises(ValueError):
            main(["--peak-tflops", "-1", "--no-plots"])

    def test_derivable_metrics_are_recomputed(self) -> None:
        row = {
            "M": "2",
            "K": "3",
            "N": "4",
            "num_gemms": "5",
            "total_flops": "240",
            "latency_ms": "2",
            "tflops": "0.00000012",
        }
        self.assertTrue(_derivable_metrics_valid(row))
        self.assertFalse(_derivable_metrics_valid({**row, "total_flops": "241"}))
        self.assertFalse(_derivable_metrics_valid({**row, "tflops": "1"}))

    def test_parallel_sweep_run_spec(self) -> None:
        size, path = parse_run_spec("16=results/custom-p16")
        self.assertEqual(size, 16)
        self.assertEqual(str(path), "results/custom-p16")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_run_spec("16")


if __name__ == "__main__":
    unittest.main()
