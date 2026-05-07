import unittest

from randopt_lora_lab.backend_next_token_probe import compare_topk, normalize_vllm_logprobs, parse_candidate_key


class FakeLogprob:
    def __init__(self, logprob, decoded_token):
        self.logprob = logprob
        self.decoded_token = decoded_token


class BackendNextTokenProbeTests(unittest.TestCase):
    def test_parse_candidate_key(self):
        candidate = parse_candidate_key("factor_gaussian_lora:seed123:s0.0075:sign-1")
        self.assertEqual(candidate.family, "factor_gaussian_lora")
        self.assertEqual(candidate.seed, 123)
        self.assertEqual(candidate.sigma, 0.0075)
        self.assertEqual(candidate.sign, -1)

    def test_normalize_vllm_logprobs_handles_objects_and_dicts(self):
        rows = normalize_vllm_logprobs(
            {
                5: FakeLogprob(-0.2, "a"),
                6: {"logprob": -0.1, "decoded_token": "b"},
            }
        )
        self.assertEqual([row["token_id"] for row in rows], [6, 5])
        self.assertEqual(rows[0]["token"], "b")

    def test_compare_topk_reports_overlap_and_deltas(self):
        left = [{"token_id": 1, "logprob": -0.1}, {"token_id": 2, "logprob": -0.4}]
        right = [{"token_id": 1, "logprob": -0.2}, {"token_id": 3, "logprob": -0.5}]
        summary = compare_topk(left, right)
        self.assertTrue(summary["top1_equal"])
        self.assertEqual(summary["topk_overlap"], 1)
        self.assertEqual(summary["topk_union"], 3)
        self.assertAlmostEqual(summary["max_common_abs_logprob_delta"], 0.1)


if __name__ == "__main__":
    unittest.main()
