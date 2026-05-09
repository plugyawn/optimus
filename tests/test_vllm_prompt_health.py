import json
import unittest
from argparse import Namespace

from randopt_lora_lab.vllm_lora_search import require_all_prompt_variants_valid_or_raise


def metrics(exact: float, *, malformed: float = 0.0, cap: float = 0.0) -> dict:
    return {
        "exact_mean": exact,
        "malformed_mean": malformed,
        "cap_hit_mean": cap,
        "answer_closed_mean": 1.0,
    }


def args() -> Namespace:
    return Namespace(max_base_malformed_for_selection=0.05, max_base_cap_hit_for_selection=0.05)


class VllmPromptHealthTests(unittest.TestCase):
    def test_all_requested_variants_valid_passes(self):
        base = {"default": metrics(0.1), "reordered": metrics(0.05)}

        require_all_prompt_variants_valid_or_raise(base, base, ["default", "reordered"], ["default", "reordered"], args())

    def test_invalid_screen_variant_raises_with_prompt_metrics(self):
        screen = {"default": metrics(0.1), "xml": metrics(0.0, malformed=0.25)}
        holdout = {"default": metrics(0.1), "xml": metrics(0.1)}

        with self.assertRaisesRegex(RuntimeError, "requested prompt variants are not all base-valid") as ctx:
            require_all_prompt_variants_valid_or_raise(screen, holdout, ["default"], ["default", "xml"], args())

        detail = json.loads(str(ctx.exception).split(": ", 1)[1])
        self.assertEqual(detail["failed"], {"screen": ["xml"]})
        self.assertEqual(detail["base_screen_by_prompt"]["xml"]["malformed_mean"], 0.25)

    def test_invalid_holdout_variant_raises_with_prompt_metrics(self):
        screen = {"default": metrics(0.1), "xml": metrics(0.1)}
        holdout = {"default": metrics(0.1), "xml": metrics(0.0, cap=0.2)}

        with self.assertRaises(RuntimeError) as ctx:
            require_all_prompt_variants_valid_or_raise(screen, holdout, ["default", "xml"], ["default"], args())

        detail = json.loads(str(ctx.exception).split(": ", 1)[1])
        self.assertEqual(detail["failed"], {"holdout": ["xml"]})
        self.assertEqual(detail["base_holdout_by_prompt"]["xml"]["cap_hit_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
