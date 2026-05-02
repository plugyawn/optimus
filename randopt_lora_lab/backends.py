from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .lora_space import Candidate, build_anzo_state, fill_lora_gaussian, lora_module_names, zero_lora


@dataclass
class GenerationResult:
    texts: list[str]
    output_tokens: int
    token_counts: list[int]
    elapsed_s: float


class StopAfterAllContain(StoppingCriteria):
    def __init__(self, prompt_len: int, stop_ids: list[int]):
        self.prompt_len = prompt_len
        self.stop_ids = torch.tensor(stop_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_ids.numel() == 0:
            return False
        generated = input_ids[:, self.prompt_len :]
        if generated.shape[1] < self.stop_ids.numel():
            return False
        stop = self.stop_ids.to(generated.device)
        for row in generated:
            found = False
            for offset in range(row.shape[0] - stop.shape[0] + 1):
                if torch.equal(row[offset : offset + stop.shape[0]], stop):
                    found = True
                    break
            if not found:
                return False
        return True


class TransformersLoraBackend:
    def __init__(
        self,
        model_name: str,
        rank: int,
        target_suffixes: tuple[str, ...],
        max_new_tokens: int,
        batch_size: int,
        dtype: str = "bf16",
        stop_at_answer: bool = False,
    ):
        self.rank = rank
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.stop_at_answer = stop_at_answer
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.answer_stop_ids = self.tokenizer("</answer>", add_special_tokens=False)["input_ids"]
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        module_names = lora_module_names(base, target_suffixes)
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=module_names,
        )
        self.model = get_peft_model(base, config)
        self.model.eval()
        zero_lora(self.model)

    def set_candidate(self, candidate: Candidate, family_state: dict | None = None):
        fill_lora_gaussian(self.model, candidate, self.rank, family_state)

    def clear_candidate(self):
        zero_lora(self.model)

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> GenerationResult:
        texts = []
        output_tokens = 0
        token_counts = []
        start = time.time()
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            stopping_criteria = None
            if self.stop_at_answer:
                stopping_criteria = StoppingCriteriaList([StopAfterAllContain(input_len, self.answer_stop_ids)])
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
            new_ids = out[:, input_len:]
            counts = (new_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            token_counts.extend(int(x) for x in counts)
            output_tokens += int(sum(counts))
            texts.extend(self.tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return GenerationResult(texts, output_tokens, token_counts, time.time() - start)

    @torch.no_grad()
    def logits_signature(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model(**inputs)
        return out.logits[:, -1, :].float().detach().cpu()

    def build_anzo_state(self, target_prompts: list[str], anchor_prompts: list[str]):
        return build_anzo_state(self.model, self.tokenizer, target_prompts, anchor_prompts, self.rank)
