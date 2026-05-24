from __future__ import annotations


VLLM_SEARCH_HELP = """\
usage: optimus vllm-search --out OUT [options]

Run a vLLM-backed LoRA perturbation search.

required:
  --out OUT                         Output directory for run files.

common options:
  --model MODEL                     Base model name or path.
  --data DATA                       Countdown JSON data path.
  --population N                    Number of candidates to screen.
  --prompts N                       Screen prompts.
  --holdout-prompts N               Heldout prompts for promoted candidates.
  --promote N                       Number of candidates to evaluate on heldout.
  --rank N                          LoRA rank.
  --sigma FLOAT                     Candidate perturbation scale.
  --targets LIST                    Comma-separated target modules.
  --tensor-parallel-size N          vLLM tensor parallel size.
  --chunk-adapters N                Candidate adapters per materialization chunk.
  --max-loras N                     Maximum active vLLM LoRA adapters.
  --max-cpu-loras N                 Maximum CPU-resident vLLM LoRA adapters.
  --max-new-tokens N                Generation token cap.
  --stop-at-answer                  Stop generation after the answer close tag.
  --antithetic                      Pair candidates by sign.
"""


VLLM_BENCH_HELP = """\
usage: optimus vllm-bench --out OUT [options]

Benchmark vLLM LoRA adapter materialization and adapter-swapping throughput.

required:
  --out OUT                         Output directory for benchmark files.

common options:
  --model MODEL                     Base model name or path.
  --data DATA                       Countdown JSON data path.
  --adapters N                      Number of candidate adapters.
  --prompts N                       Number of prompts.
  --rank N                          LoRA rank.
  --sigma FLOAT                     Candidate perturbation scale.
  --targets LIST                    Comma-separated target modules.
  --tensor-parallel-size N          vLLM tensor parallel size.
  --max-loras N                     Maximum active vLLM LoRA adapters.
  --max-cpu-loras N                 Maximum CPU-resident vLLM LoRA adapters.
  --preload                         Preload adapters before benchmark generation.
  --mixed-batch                     Issue mixed-adapter batches.
  --prepare-only                    Materialize adapters without starting vLLM.
"""


VLLM_HALVING_HELP = """\
usage: optimus vllm-halving --out OUT [options]

Run staged vLLM-backed LoRA perturbation screening and heldout confirmation.

required:
  --out OUT                         Output directory for staged-search files.

common options:
  --model MODEL                     Base model name or path.
  --data DATA                       Countdown JSON data path.
  --population N                    Number of candidates to screen.
  --stage-prompts N                 Prompts used in the first screening stage.
  --survivors N                     Candidates retained after the first stage.
  --prompts N                       Full screen prompts.
  --holdout-prompts N               Heldout prompts for promoted candidates.
  --promote N                       Number of candidates to evaluate on heldout.
  --rank N                          LoRA rank.
  --sigma FLOAT                     Candidate perturbation scale.
  --targets LIST                    Comma-separated target modules.
  --tensor-parallel-size N          vLLM tensor parallel size.
  --chunk-adapters N                Candidate adapters per materialization chunk.
  --antithetic                      Pair candidates by sign.
"""


PEFT_SEARCH_HELP = """\
usage: optimus peft-search --out OUT [options]

Run a trusted Transformers search with either LoRA adapters or dense in-memory perturbations.

required:
  --out OUT                         Output directory for run files.

common options:
  --model MODEL                     Base model name or path.
  --data DATA                       Countdown JSON data path.
  --population N                    Number of candidates to screen.
  --prompts N                       Screen prompts.
  --holdout-prompts N               Heldout prompts for promoted candidates.
  --promote N                       Number of candidates to evaluate on heldout.
  --rank N                          LoRA rank.
  --sigma FLOAT                     Candidate perturbation scale.
  --targets LIST                    Comma-separated target modules.
  --perturbation-backend {lora,dense}
                                    Materialize candidates as LoRA adapters or dense patches.
  --batch-size N                    HF/PEFT generation batch size.
  --max-new-tokens N                Generation token cap.
  --stop-at-answer                  Stop generation after the answer close tag.
  --antithetic                      Pair candidates by sign.
"""
