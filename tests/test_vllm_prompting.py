import unittest

from optimus.serving.prompting import make_vllm_prompt_inputs
from randopt_lora_lab.vllm_prompting import make_vllm_prompt_inputs as legacy_make_vllm_prompt_inputs


class FakeTokenizer:
    bos_token_id = 98

    def __call__(self, text, *, add_special_tokens=True, **_kwargs):
        ids = [ord(ch) % 251 for ch in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def prompt_ids(prompt):
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, dict):
        return list(prompt["prompt_token_ids"])
    return list(prompt.prompt_token_ids)


class VllmPromptingTests(unittest.TestCase):
    def test_legacy_namespace_reexports_public_prompt_input_builder(self):
        self.assertIs(legacy_make_vllm_prompt_inputs, make_vllm_prompt_inputs)

    def test_text_mode_preserves_prompt_strings(self):
        prompts = ["ab", "cd"]

        self.assertEqual(make_vllm_prompt_inputs(prompts, FakeTokenizer(), "text"), prompts)

    def test_token_id_mode_tokenizes_with_special_tokens(self):
        prompts = make_vllm_prompt_inputs(["ab"], FakeTokenizer(), "token_ids")

        self.assertEqual(prompt_ids(prompts[0]), [98, ord("a") % 251, ord("b") % 251])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            make_vllm_prompt_inputs(["ab"], FakeTokenizer(), "missing")


if __name__ == "__main__":
    unittest.main()
