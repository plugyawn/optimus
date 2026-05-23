import unittest
import types

from optimus.serving.contracts import (
    backend_contract,
    contract_max_tokens,
    sampling_params_contract,
    tokenizer_contract,
    vllm_tokenizer_contract,
)
from randopt_lora_lab.backend_contract import backend_contract as legacy_backend_contract


class FakeTokenizer:
    padding_side = "left"
    pad_token = "<eos>"
    pad_token_id = 99
    eos_token = "<eos>"
    eos_token_id = 99
    bos_token = "<bos>"
    bos_token_id = 98
    unk_token = "<unk>"
    unk_token_id = 0
    chat_template = "fake-template"

    def __call__(self, text, *, add_special_tokens=True, **_kwargs):
        ids = [ord(ch) % 251 for ch in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def decode(self, ids):
        return "|".join(str(token_id) for token_id in ids)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")
        self.stop = kwargs.get("stop")
        self.include_stop_str_in_output = kwargs.get("include_stop_str_in_output", False)
        self.ignore_eos = False
        self.skip_special_tokens = True


class OldFakeSamplingParams:
    def __init__(self, **kwargs):
        if "include_stop_str_in_output" in kwargs:
            raise TypeError("old vllm")
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")
        self.stop = kwargs.get("stop")


class FakeLLM:
    def get_tokenizer(self):
        return FakeTokenizer()


class NoTokenizerLLM:
    pass


class BackendContractTests(unittest.TestCase):
    def test_legacy_namespace_reexports_public_backend_contract(self):
        self.assertIs(legacy_backend_contract, backend_contract)

    def test_tokenizer_contract_records_exact_prompt_ids_and_stop_ids(self):
        contract = tokenizer_contract(FakeTokenizer(), ["ab", "cd"])

        self.assertEqual(contract["padding_side"], "left")
        self.assertEqual(contract["answer_stop_text"], "</answer>")
        self.assertEqual(contract["answer_stop_ids"], [ord(ch) % 251 for ch in "</answer>"])
        self.assertEqual(contract["prompt_count"], 2)
        self.assertEqual(contract["prompts"][0]["token_ids"], [98, ord("a") % 251, ord("b") % 251])
        self.assertEqual(contract["min_prompt_tokens"], 3)
        self.assertEqual(contract["max_prompt_tokens"], 3)
        self.assertTrue(contract["prompt_token_ids_sha256"])

    def test_sampling_contract_records_stop_kwargs(self):
        contract = sampling_params_contract(FakeSamplingParams, 32, True)

        self.assertEqual(contract["requested_kwargs"]["max_tokens"], 32)
        self.assertEqual(contract["requested_kwargs"]["stop"], ["</answer>"])
        self.assertEqual(contract["actual_attrs"]["include_stop_str_in_output"], True)
        self.assertFalse(contract["dropped_include_stop_str_in_output"])

    def test_sampling_contract_handles_old_vllm_without_include_stop(self):
        contract = sampling_params_contract(OldFakeSamplingParams, 16, True)

        self.assertTrue(contract["dropped_include_stop_str_in_output"])
        self.assertNotIn("include_stop_str_in_output", contract["used_kwargs"])
        self.assertEqual(contract["actual_attrs"]["stop"], ["</answer>"])

    def test_backend_contract_combines_tokenizer_and_sampling(self):
        args = types.SimpleNamespace(
            model="fake-model",
            max_new_tokens=7,
            stop_at_answer=True,
            hf_batch_size=2,
            hf_dtype="bf16",
            vllm_dtype="bfloat16",
        )

        contract = backend_contract(FakeTokenizer(), ["xy"], args, FakeSamplingParams)

        self.assertEqual(contract["kind"], "backend_prompt_decode_contract")
        self.assertEqual(contract["model"], "fake-model")
        self.assertEqual(contract["tokenizer"]["prompts"][0]["token_ids"], [98, ord("x") % 251, ord("y") % 251])
        self.assertEqual(contract["vllm_sampling"]["actual_attrs"]["max_tokens"], 7)

    def test_backend_contract_defaults_one_token_for_next_token_probes(self):
        args = types.SimpleNamespace(model="fake-model", stop_at_answer=False)

        contract = backend_contract(FakeTokenizer(), ["xy"], args, FakeSamplingParams)

        self.assertEqual(contract_max_tokens(args), 1)
        self.assertEqual(contract["max_new_tokens"], 1)
        self.assertEqual(contract["vllm_sampling"]["actual_attrs"]["max_tokens"], 1)

    def test_vllm_tokenizer_contract_records_available_tokenizer(self):
        contract = vllm_tokenizer_contract(FakeLLM(), ["zz"])

        self.assertTrue(contract["available"])
        self.assertEqual(contract["prompts"][0]["token_ids"], [98, ord("z") % 251, ord("z") % 251])

    def test_vllm_tokenizer_contract_records_unavailable_tokenizer(self):
        contract = vllm_tokenizer_contract(NoTokenizerLLM(), ["zz"])

        self.assertFalse(contract["available"])
        self.assertIn("reason", contract)


if __name__ == "__main__":
    unittest.main()
