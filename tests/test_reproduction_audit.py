import unittest

from randopt_lora_lab.reproduction_audit import audit_official_countdown_run, official_countdown_ensemble_ks, summarize


def official_summary():
    return {
        "model": "allenai/Olmo-3-7B-Instruct",
        "perturbation_backend": "dense",
        "family": "dense_gaussian",
        "targets": "all_params",
        "dense_noise_mode": "paper",
        "screen_prompts": 200,
        "holdout_prompts": 1000,
        "population": 5000,
        "sigma_values": [0.0005, 0.001, 0.002],
        "max_new_tokens": 1024,
        "prompt_variant": "paper",
        "use_chat_template": True,
        "ensemble_ks": [50, 200, 250, 500],
        "screen_holdout_overlap": 0,
        "screen_unique_semantic_prompts": 200,
        "holdout_unique_semantic_prompts": 1000,
        "ensemble_holdout": [{"k": 50, "exact_mean": 0.1}],
    }


class ReproductionAuditTests(unittest.TestCase):
    def test_official_ensemble_ks_come_from_top_k_ratios(self):
        self.assertEqual(official_countdown_ensemble_ks(5000), [50, 200, 250, 500])
        self.assertEqual(official_countdown_ensemble_ks(128), [1, 5, 6, 12])

    def test_official_summary_passes(self):
        summary = summarize(audit_official_countdown_run(official_summary()))

        self.assertTrue(summary["pass"])
        self.assertEqual(summary["failed"], [])

    def test_answer_only_qv_panel_fails_reproduction(self):
        row = official_summary()
        row.update(
            {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "targets": "q_proj,v_proj",
                "dense_noise_mode": "canonical",
                "screen_prompts": 64,
                "population": 128,
                "max_new_tokens": 128,
                "prompt_variant": "default",
                "use_chat_template": False,
                "ensemble_ks": [8, 16, 32],
                "screen_unique_semantic_prompts": 64,
            }
        )

        summary = summarize(audit_official_countdown_run(row))

        self.assertFalse(summary["pass"])
        self.assertIn("model", summary["failed"])
        self.assertIn("full_parameter_targets", summary["failed"])
        self.assertIn("prompt_variant", summary["failed"])
        self.assertIn("ensemble_ks", summary["failed"])


if __name__ == "__main__":
    unittest.main()
