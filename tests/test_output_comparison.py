"""Unit tests for output comparison logic."""

import json
import unittest
from benchmark import BenchmarkResult, compare_outputs


class TestOutputComparison(unittest.TestCase):
    def test_identical_outputs(self):
        baseline = BenchmarkResult(
            strategy="dynamic",
            status="success",
            output_token_ids=json.dumps([10, 20, 30, 40, 50]),
            output_text="The quick brown fox",
        )
        optimized = BenchmarkResult(
            strategy="quantized_quanto_4bit",
            status="success",
            output_token_ids=json.dumps([10, 20, 30, 40, 50]),
            output_text="The quick brown fox",
        )
        compare_outputs(baseline, optimized)

        self.assertEqual(optimized.match_ratio, 1.0)
        self.assertEqual(optimized.prefix_match_length, 5)
        self.assertTrue(optimized.full_match)

    def test_partial_match(self):
        baseline = BenchmarkResult(
            strategy="dynamic",
            status="success",
            output_token_ids=json.dumps([10, 20, 30, 40, 50]),
        )
        optimized = BenchmarkResult(
            strategy="quantized_quanto_2bit",
            status="success",
            output_token_ids=json.dumps([10, 20, 99, 40, 88]),
        )
        compare_outputs(baseline, optimized)

        self.assertEqual(optimized.prefix_match_length, 2)
        # Matches: indices 0, 1, 3 match -> 3 / 5 = 0.6
        self.assertAlmostEqual(optimized.match_ratio, 0.6)
        self.assertFalse(optimized.full_match)

    def test_divergent_lengths(self):
        baseline = BenchmarkResult(
            strategy="dynamic",
            status="success",
            output_token_ids=json.dumps([10, 20, 30, 40]),
        )
        optimized = BenchmarkResult(
            strategy="quantized_quanto_2bit",
            status="success",
            output_token_ids=json.dumps([10, 20]),
        )
        compare_outputs(baseline, optimized)

        self.assertEqual(optimized.prefix_match_length, 2)
        # 2 matches out of max_len 4 = 0.5
        self.assertAlmostEqual(optimized.match_ratio, 0.5)
        self.assertFalse(optimized.full_match)

    def test_empty_tokens(self):
        baseline = BenchmarkResult(
            strategy="dynamic",
            status="success",
            output_token_ids="[]",
        )
        optimized = BenchmarkResult(
            strategy="quantized_quanto_4bit",
            status="success",
            output_token_ids="[]",
        )
        compare_outputs(baseline, optimized)

        self.assertEqual(optimized.match_ratio, 1.0)
        self.assertEqual(optimized.prefix_match_length, 0)
        self.assertTrue(optimized.full_match)

    def test_skipped_on_failed_status(self):
        baseline = BenchmarkResult(strategy="dynamic", status="error")
        optimized = BenchmarkResult(strategy="quantized_quanto_4bit", status="success",
                                    output_token_ids=json.dumps([1, 2, 3]))
        compare_outputs(baseline, optimized)

        self.assertEqual(optimized.match_ratio, -1)  # Unmodified


if __name__ == "__main__":
    unittest.main()

