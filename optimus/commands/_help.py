from __future__ import annotations


VLLM_SEARCH_HELP = """\
usage: optimus vllm-search --out OUT [options]

Run a vLLM LoRA candidate search.

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

advanced options are preserved for compatibility with existing run manifests.
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

Run staged candidate screening and heldout confirmation.

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

Run a trusted HF/PEFT LoRA candidate search.

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
  --batch-size N                    HF/PEFT generation batch size.
  --max-new-tokens N                Generation token cap.
  --stop-at-answer                  Stop generation after the answer close tag.
  --antithetic                      Pair candidates by sign.
"""


AGGREGATE_LORA_HELP = """\
usage: optimus aggregate-lora --source-run SOURCE_RUN --out OUT [options]

Evaluate a serveable aggregate of top LoRA perturbations.

required:
  --source-run SOURCE_RUN           Source run with candidate_summary.jsonl.
  --out OUT                         Output directory for aggregate evaluation files.

common options:
  --model MODEL                     Base model name or path.
  --data DATA                       Countdown JSON data path.
  --prompts N                       Screen prompts.
  --holdout-prompts N               Heldout prompts.
  --base-rank N                     Rank of each source candidate.
  --top-k N                         Number of source candidates to aggregate.
  --weight-mode MODE                uniform, score, or centered.
  --targets LIST                    Comma-separated target modules.
  --max-new-tokens N                Generation token cap.
  --batch-size N                    HF/PEFT generation batch size.
  --stop-at-answer                  Stop generation after the answer close tag.
"""
