import unittest

from randopt_lora_lab.countdown import CountdownExample
from randopt_lora_lab.prompt_variants import make_variant_prompts, prompt_fn


class PromptVariantsTests(unittest.TestCase):
    def test_reordered_preserves_countdown_contract(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)
        text = make_variant_prompts([ex], "reordered")[0]

        self.assertIn("Numbers: 3, 7, 8, 8", text)
        self.assertIn("Target: 24", text)
        self.assertIn("exactly once", text)
        self.assertIn("<answer>", text)
        self.assertIn("</answer>", text)
        self.assertIn("Do not include an equals sign", text)

    def test_unknown_prompt_variant_rejected(self):
        with self.assertRaises(ValueError):
            prompt_fn("missing")


if __name__ == "__main__":
    unittest.main()
