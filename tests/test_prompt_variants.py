import unittest

from optimus.tasks.countdown import CountdownExample
from optimus.tasks.prompt_variants import UPSTREAM_SYSTEM_MESSAGE, make_variant_prompts, prompt_fn
from randopt_lora_lab.prompt_variants import make_variant_prompts as legacy_make_variant_prompts
from randopt_lora_lab.prompt_variants import prompt_fn as legacy_prompt_fn


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = "\n".join(f"{row['role']}:{row['content']}" for row in messages)
        return rendered + "\nassistant:"


class PromptVariantsTests(unittest.TestCase):
    def test_legacy_namespace_delegates_default_prompt_fn(self):
        self.assertIs(legacy_prompt_fn("default"), prompt_fn("default"))

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

    def test_upstream_variant_matches_reasoning_contract(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)
        text = make_variant_prompts([ex], "upstream")[0]

        self.assertTrue(text.startswith(UPSTREAM_SYSTEM_MESSAGE))
        self.assertIn("Using the numbers [3, 7, 8, 8]", text)
        self.assertIn("Show your work in <think> </think> tags", text)
        self.assertIn("<answer> </answer>", text)

    def test_upstream_variant_can_use_chat_template(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)
        text = make_variant_prompts([ex], "upstream", tokenizer=FakeTokenizer(), use_chat_template=True)[0]

        self.assertIn("system:" + UPSTREAM_SYSTEM_MESSAGE, text)
        self.assertIn("user:Using the numbers [3, 7, 8, 8]", text)
        self.assertTrue(text.endswith("assistant:"))

    def test_legacy_prompt_name_aliases_upstream_reasoning_contract(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)
        paper = legacy_make_variant_prompts([ex], "paper")[0]
        upstream = make_variant_prompts([ex], "upstream")[0]

        self.assertEqual(upstream, paper)


if __name__ == "__main__":
    unittest.main()
