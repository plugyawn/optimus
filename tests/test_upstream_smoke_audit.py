import unittest

from randopt_lora_lab.upstream_smoke_audit import audit_upstream_countdown_smoke


def upstream_payload(population: int = 32):
    return {
        "args": {
            "dataset": "countdown",
            "model_name": "allenai/Olmo-3-7B-Instruct",
            "train_data_path": "data/countdown/countdown_official.json",
            "test_data_path": "data/countdown/countdown_official.json",
            "train_samples": 200,
            "test_samples": 128,
            "population_size": population,
            "sigma_list": [0.0005, 0.001, 0.002],
            "top_k_ratios": "0.04,0.01,0.05,0.1",
            "top_k_list": [3, 1] if population == 32 else [200, 50, 250, 500],
            "max_tokens": 1024,
        },
        "results": {
            "base_train_accuracy": 0.72715,
            "base_test_accuracy": 0.67586,
            "top_k_perturbs": [[952224740, 0.001]],
            "top_k_train_rewards": [0.7942],
            "ensemble_results": {"1": {"accuracy": 64.8}, "3": {"accuracy": 71.1}},
        },
        "top_k_seeds": {
            "top_k_models": [
                {"rank": 1, "seed": 952224740, "sigma": 0.001, "train_reward": 0.7942},
            ]
        },
    }


class UpstreamSmokeAuditTests(unittest.TestCase):
    def test_reduced_population_official_semantics_passes_smoke_not_paper_scale(self):
        summary = audit_upstream_countdown_smoke(upstream_payload(), min_population=32, min_test_samples=128)

        self.assertTrue(summary["smoke_pass"])
        self.assertFalse(summary["paper_scale_pass"])
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["failed"], [])

    def test_require_paper_scale_fails_for_p32(self):
        summary = audit_upstream_countdown_smoke(
            upstream_payload(),
            require_paper_scale=True,
            min_population=32,
            min_test_samples=128,
        )

        self.assertFalse(summary["pass"])
        self.assertTrue(summary["smoke_pass"])
        self.assertIn("paper_scale_population", summary["failed"])

    def test_local_like_payload_fails_official_semantics(self):
        payload = upstream_payload()
        payload["args"].update(
            {
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "train_data_path": "data/countdown_generated_1200_seed20260507.json",
                "population_size": 128,
                "top_k_list": [8, 16, 32],
                "max_tokens": 128,
            }
        )

        summary = audit_upstream_countdown_smoke(payload, min_population=32)

        self.assertFalse(summary["smoke_pass"])
        self.assertIn("model", summary["failed"])
        self.assertIn("official_train_data", summary["failed"])
        self.assertIn("top_k_list", summary["failed"])
        self.assertIn("max_tokens", summary["failed"])


if __name__ == "__main__":
    unittest.main()
