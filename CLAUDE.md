# CLAUDE.md

## Project Goal

This project is a minimal TRL GRPO training scaffold for interleaved reasoning on
`nvidia/Nemotron-CrossThink`. The policy should produce structured completions with
private reasoning in `<think>...</think>` and the final result in
`<answer>...</answer>`.

The intended direction is to combine:

- Local training flow from `train.py`.
- Reference implementation patterns from `../reasoning_grpo/interleaved_grpo/train.py`.
- Dataset normalization for Hugging Face `nvidia/Nemotron-CrossThink`.
- Reward behavior inspired by NVIDIA NeMo RL.

## Current Layout

This repo is intentionally flat right now:

- `train.py`: CLI entry point using `trl.GRPOTrainer`, LoRA via PEFT, and two local reward functions.
- `datasets_loader.py`: dataset normalization for `gsm8k`, `math`, and `nemotron-crossthink`.
- `test_crossthink.py`: small CrossThink loading script, but it currently imports
  `interleaved_grpo.datasets_loader`, which does not exist in this repo.

There is no local package directory and no `requirements.txt` yet. Do not assume the
`../reasoning_grpo` package structure exists here unless you create it deliberately.

## External References

Use these as source references when extending the project:

- Neighbor reference: `../reasoning_grpo/interleaved_grpo/train.py`
  - Has more robust TRL-version compatibility, prompt-length filtering, reward weights,
    generation JSONL logging, `processing_class=tokenizer`, optional vLLM, optional DAPO,
    and safer LoRA handling through `peft_config` when supported.
- Dataset: `https://huggingface.co/datasets/nvidia/Nemotron-CrossThink`
  - The math data used here comes from `Data/Nemotron-CrossThink-Math.jsonl`.
  - Public dataset fields include `data_source`, `prompt`, `reward_model`, and `meta_data`.
  - Current loader extracts `meta_data["question"]` and `reward_model["ground_truth"]`.
- NeMo RL reward source: `https://github.com/NVIDIA-NeMo/RL`
  - `nemo_rl/environments/rewards.py` contains reward combinators and helpers such as
    math expression reward, exact alphanumeric answer reward, format reward, and reward
    function combination/weighting.
  - `nemo_rl/environments/math_environment.py` uses `math_verify`-based parsing and
    verification for math answers.

## Dataset Notes

`datasets_loader.py` normalizes each row to:

- `question`: the math problem prompt.
- `answer`: the target final answer.
- `task_type`: currently always `"math"` for CrossThink.

For `nemotron-crossthink`, use `--dataset-split train`. The loader reads
`hf://datasets/nvidia/Nemotron-CrossThink/Data/Nemotron-CrossThink-Math.jsonl`
directly because loading the default dataset config can infer a narrower schema from
the QA split and then fail on math rows that include extra `meta_data` fields.

Keep CrossThink processing separate from `gsm8k` and `hendrycks/competition_math`; the
CrossThink source rows are nested dictionaries, not simple `question`/`answer` rows.

## Training Entry Point

Basic command shape after dependencies are installed:

```bash
python train.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset nemotron-crossthink \
  --dataset-split train \
  --max-samples 16 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-generations 2 \
  --output-dir ./nemo_rl_output_smoke
```

`train.py` currently loads the model directly with
`AutoModelForCausalLM.from_pretrained(..., device_map="auto")`, applies LoRA by default,
and passes the model object to `GRPOTrainer`.

To force private reasoning in a specific language, pass `--reasoning-lang`, for example:

```bash
python train.py \
  --dataset nemotron-crossthink \
  --reasoning-lang yue \
  --max-samples 16 \
  --max-steps 1
```

This updates the chat prompt to request `<think>` reasoning in the target language and
adds a language consistency reward with default multiplier `--language-reward-weight 0.2`.

Generated completions are logged to JSONL by default at `OUTPUT_DIR/generations.jsonl`.
Use `--generation-log-file path/to/file.jsonl` to choose a path, or
`--disable-generation-logging` to turn it off. The logger is implemented as a no-op
reward function that always returns `0.0`, so it does not change training scores.

Use `--wandb` or `-wandb` to report TRL training metrics to Weights & Biases. The
underlying `report_to` value is resolved from `--report-to` plus the W&B convenience flag.

## Reward Functions

Current local rewards are intentionally simple:

- `xml_format_reward`: partial reward for containing `<think>` and `<answer>` tags.
- `correctness_reward`: extracts the first `<answer>...</answer>` span and requires exact
  string equality with the normalized target.
- `language_consistency_reward`: enabled when `--reasoning-lang` is set or when the
  dataset has `reasoning_lang` / `reasoning_language` / `language` metadata. It checks
  text inside `<think>` blocks against `yue`, `zh`, or `en`, using the
  `../reasoning_grpo` reward as the reference.

Auxiliary rewards are correctness-gated. If `correctness_reward` returns `0.0` for a
completion, the format and language reward functions also return `0.0`, so the final
summed GRPO reward is zero for incorrect answers.

When improving this, prefer the NeMo RL pattern:

- Use a weighted combination of independent reward functions.
- Keep format reward separate from correctness reward.
- Use `math_verify`-style parsing/equivalence for mathematical targets instead of exact
  string equality.
- Return one float per completion and keep reward signatures compatible with TRL custom
  rewards: `reward_fn(completions, **dataset_columns)` or
  `reward_fn(prompts, completions, **dataset_columns)`, depending on the trainer version
  and wrapper style.

## Known Caveats

- `test_crossthink.py` is stale for this flat repo because it imports
  `interleaved_grpo.datasets_loader`. Change it to import local `datasets_loader` before
  relying on it.
- `--bf16` and `--use-lora` are declared with `action="store_true", default=True`, so they
  are always enabled from the CLI. Use `argparse.BooleanOptionalAction` if `--no-bf16` or
  `--no-use-lora` should be possible.
- Local exact-answer matching is too strict for math training. CrossThink answers may need
  normalization or symbolic equivalence.
- There is no dependency file. At minimum the runtime needs `datasets`, `trl`, `peft`,
  `transformers`, and `torch`; NeMo-style math rewards also need `math-verify`.
- Current environment check found `transformers` and `torch`, but not `datasets`, `trl`,
  `peft`, or `math_verify`.
- Do not launch a full 7B GRPO training run as validation. Start with compile checks,
  dataset loading, and a tiny `--max-steps 1` smoke run.

## Validation

Lightweight checks that do not require downloading a model:

```bash
python -m compileall train.py datasets_loader.py test_crossthink.py
python - <<'PY'
from datasets_loader import build_interleaved_messages
print(build_interleaved_messages("What is 2+2?"))
PY
```

After installing `datasets`, smoke-test CrossThink loading:

```bash
python - <<'PY'
from datasets_loader import load_training_dataset
ds = load_training_dataset("nemotron-crossthink", split="train", max_samples=2)
print(len(ds), ds.column_names)
print(ds[0])
PY
```

Reward validation against an 8-row CrossThink sample confirmed that exact gold
`<answer>` completions score `1.8` with `--reasoning-lang en`, while wrong but
well-formatted completions and missing-answer completions both score `0.0` after the
correctness gate.

Only after dataset loading and reward behavior are verified, run a one-step GRPO smoke run
with a small sample count and a writable output directory.
