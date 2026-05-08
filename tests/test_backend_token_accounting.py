import unittest

from randopt_lora_lab.backends import visible_token_count
from randopt_lora_lab.cap_stability import metric_row


class BackendTokenAccountingTests(unittest.TestCase):
    def test_visible_token_count_stops_at_first_answer_close(self):
        self.assertEqual(visible_token_count([10, 11, 12, 13, 14, 15], [12, 13]), 4)

    def test_visible_token_count_uses_full_length_without_stop(self):
        self.assertEqual(visible_token_count([10, 11, 12], [98, 99]), 3)
        self.assertEqual(visible_token_count([10, 11, 12], []), 3)

    def test_metric_row_preserves_raw_output_tokens_when_present(self):
        ev = {
            "candidate": "c",
            "exact_mean": 0.25,
            "malformed_mean": 0.0,
            "cap_hit_mean": 1.0,
            "answer_closed_mean": 1.0,
            "output_tokens": 4,
            "raw_output_tokens": 128,
            "output_token_mean": 4.0,
            "output_token_p95": 4.0,
            "elapsed_s": 1.0,
            "mutation_s": 0.0,
        }

        row = metric_row(ev, cap=128, prompt_variant="default", split="screen", candidate_kind="base")

        self.assertEqual(row["output_tokens"], 4)
        self.assertEqual(row["raw_output_tokens"], 128)
        self.assertEqual(row["cap_hit_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
