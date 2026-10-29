"""Unit tests for memory_estimator.py."""

import unittest
from types import SimpleNamespace
from memory_estimator import estimate_kv_cache_memory, get_kv_cache_params, KVCacheEstimate


class TestMemoryEstimator(unittest.TestCase):
    def setUp(self):
        # Mock Qwen2.5-0.5B config: 24 layers, 14 attention heads, 2 KV heads (GQA), head_dim=64, hidden_size=896
        self.mock_gqa_config = SimpleNamespace(
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            hidden_size=896,
        )

        # Mock standard MHA config (no num_key_value_heads, derived head_dim)
        self.mock_mha_config = SimpleNamespace(
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
        )

    def test_gqa_params_extraction(self):
        layers, kv_heads, head_dim = get_kv_cache_params(self.mock_gqa_config)
        self.assertEqual(layers, 24)
        self.assertEqual(kv_heads, 2)  # Should use num_key_value_heads, not num_attention_heads
        self.assertEqual(head_dim, 64)

    def test_mha_params_extraction(self):
        layers, kv_heads, head_dim = get_kv_cache_params(self.mock_mha_config)
        self.assertEqual(layers, 12)
        self.assertEqual(kv_heads, 12)  # Should fall back to num_attention_heads
        self.assertEqual(head_dim, 64)  # 768 // 12 = 64

    def test_estimate_formula_float16(self):
        # 2 * 24 * 2 * 64 * 128 * 1 * 2 = 1,572,864 bytes = 1.5 MB
        est = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=128, batch_size=1, dtype="float16")
        expected_bytes = 2 * 24 * 2 * 64 * 128 * 1 * 2
        self.assertEqual(est.total_bytes, expected_bytes)
        self.assertAlmostEqual(est.total_mb, 1.5, places=4)

    def test_batch_size_scaling(self):
        est_bs1 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=256, batch_size=1, dtype="float16")
        est_bs4 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=256, batch_size=4, dtype="float16")
        self.assertEqual(est_bs4.total_bytes, est_bs1.total_bytes * 4)

    def test_sequence_length_scaling(self):
        est_128 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=128, batch_size=1, dtype="float16")
        est_512 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=512, batch_size=1, dtype="float16")
        self.assertEqual(est_512.total_bytes, est_128.total_bytes * 4)

    def test_quantized_dtypes(self):
        est_fp16 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=256, batch_size=1, dtype="float16")
        est_int8 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=256, batch_size=1, dtype="int8")
        est_int4 = estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=256, batch_size=1, dtype="int4")

        # INT8 should be half of FP16
        self.assertEqual(est_int8.total_bytes, est_fp16.total_bytes // 2)
        # INT4 should be one-fourth of FP16
        self.assertEqual(est_int4.total_bytes, est_fp16.total_bytes // 4)

    def test_invalid_dtype_raises_error(self):
        with self.assertRaises(ValueError):
            estimate_kv_cache_memory(self.mock_gqa_config, sequence_length=128, dtype="unsupported_dtype")

    def test_unit_conversions(self):
        est = KVCacheEstimate(
            num_layers=1, num_kv_heads=1, head_dim=1, sequence_length=1,
            batch_size=1, bytes_per_element=1, dtype="int8", total_bytes=1024 * 1024 * 1024
        )
        self.assertAlmostEqual(est.total_mb, 1024.0)
        self.assertAlmostEqual(est.total_gb, 1.0)


if __name__ == "__main__":
    unittest.main()

