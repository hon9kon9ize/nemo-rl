# NeMo RL CrossThink GRPO

Minimal TRL GRPO training scaffold for interleaved reasoning on
`nvidia/Nemotron-CrossThink`.

The model is prompted to place private reasoning in `<think>...</think>` and the
final answer in `<answer>...</answer>`. Auxiliary rewards are correctness-gated:
if the final answer is wrong, format and language rewards are forced to `0.0`.

## Files

- `train.py`: GRPO training entry point, reward functions, W&B support, and JSONL generation logging.
- `datasets_loader.py`: dataset loading and normalization for GSM8K, MATH, and Nemotron CrossThink math data.
- `run_pjm_train.sh`: PJM batch launcher for one-GPU training.
- `CLAUDE.md`: implementation notes and caveats for coding agents.

## Dependencies

Install the runtime packages in your training environment:

```bash
python -m pip install datasets transformers torch trl peft
```

Optional:

```bash
python -m pip install wandb cantofilter
```

`cantofilter` improves Cantonese detection for `--reasoning-lang yue`; the code has a
lightweight fallback when it is not installed.

## Dataset

For `--dataset nemotron-crossthink`, the loader reads the CrossThink math JSONL directly:

```text
hf://datasets/nvidia/Nemotron-CrossThink/Data/Nemotron-CrossThink-Math.jsonl
```

Rows are normalized to:

- `question`
- `answer`
- `task_type="math"`

Use `--dataset-split train`; `train_math` is also accepted as an alias.

## Reward Behavior

Default reward functions:

1. `correctness_reward`: exact match against text inside `<answer>...</answer>`.
2. `correctness_gated_xml_format_reward`: format score, but only when correctness is `1.0`.
3. `correctness_gated_language_consistency_reward`: optional language score, enabled by `--reasoning-lang`, also gated by correctness.
4. `generation_jsonl_logger`: no-op reward returning `0.0`; logs generations for debugging.

The trainer follows the `../reasoning_grpo` chat-template handling:

- `--assistant-prefill-think` is enabled by default and appends `<think>\n` after the
  assistant generation marker when the tokenizer template does not already do so.
- `--normalize-prefilled-think` is enabled by default and reconstructs that prefilled
  opening tag before reward parsing, so completions that start with `</think>` can still
  be scored as balanced `<think>...</think>` blocks.
- `--chat-template-enable-thinking` / `--no-chat-template-enable-thinking` can pass an
  explicit `enable_thinking` value to tokenizer chat templates that support it.

Before model loading, `train.py` validates the formatted `prompt` column and prints the
assistant-generation tail for a few rows. For Qwen3-style reasoning runs, the prompt tail
should end with an unmatched assistant-side `<think>` prefix, for example:

```text
<|im_start|>assistant
<think>
```

Run prompt validation without starting training:

```bash
python train.py \
  --model-id /path/to/model_or_tokenizer \
  --dataset nemotron-crossthink \
  --max-samples 2 \
  --reasoning-lang yue \
  --validate-prompts-only
```

With `--reasoning-lang en`, validated CrossThink examples score:

```text
correct gold answer:        [1.0, 0.6, 0.2, 0.0] sum=1.8
wrong but valid format:     [0.0, 0.0, 0.0, 0.0] sum=0.0
missing <answer>:           [0.0, 0.0, 0.0, 0.0] sum=0.0
```

## Smoke Checks

Run lightweight checks before any model training:

```bash
python -m compileall train.py datasets_loader.py test_crossthink.py
python train.py --help
```

Dataset smoke test:

```bash
python - <<'PY'
from datasets_loader import load_training_dataset

ds = load_training_dataset("nemotron-crossthink", split="train", max_samples=2)
print(len(ds), ds.column_names)
print(ds[0])
PY
```

Reward smoke test:

```bash
python - <<'PY'
from train import build_parser, select_reward_funcs

args = build_parser().parse_args(["--reasoning-lang", "en"])
funcs = select_reward_funcs(args, {"column_names": ["answer", "reasoning_lang"]})
completion = ["<think>We can add two and two.</think><answer>4</answer>"]
kwargs = {"answer": ["4"], "reasoning_lang": ["en"], "task_type": ["math"]}
print([fn(completion, **kwargs) for fn in funcs])
PY
```

## Local Training

Tiny trainer smoke run:

```bash
python -u train.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset nemotron-crossthink \
  --dataset-split train \
  --max-samples 16 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-generations 2 \
  --reasoning-lang yue \
  --output-dir ./nemo_rl_output_smoke
```

## Generation Logs

Generated completions are logged by default:

```text
OUTPUT_DIR/generations.jsonl
```

Override or disable logging:

```bash
python train.py --generation-log-file ./runs/debug_generations.jsonl
python train.py --disable-generation-logging
python train.py --no-generation-log-prompts
```

Each JSONL record includes the completion, target answer, reasoning language, reward
components, weighted reward, process rank, and prompt when available.

## Weights & Biases

Enable W&B metrics:

```bash
python train.py --wandb
```

The single-dash alias is also supported:

```bash
python train.py -wandb
```

Set the project name with:

```bash
export WANDB_PROJECT=nemo-rl-crossthink
```

## PJM Batch Run

Submit or run the launcher:

```bash
bash run_pjm_train.sh
```

Common overrides:

```bash
PROJECT_DIR=/path/to/project \
MODEL=/path/to/merged_model \
OUTPUT_DIR=./nemo_rl_output/crossthink_yue_lora \
MAX_SAMPLES=10000 \
MAX_PROMPT=2048 \
MAX_COMPLETION=1024 \
bash run_pjm_train.sh
```

The launcher enables LoRA, Cantonese reasoning (`--reasoning-lang yue`), W&B, and
generation logging to `$OUTPUT_DIR/generations.jsonl`.

## Caveats

- `--bf16` and `--use-lora` currently default to enabled because they use
  `store_true` with `default=True`; there is no `--no-bf16` or `--no-use-lora` yet.
- `train.py` filters `GRPOConfig` kwargs against the installed TRL version. Unsupported
  fields such as `max_prompt_length` are printed and ignored instead of crashing.
- Math correctness is exact string matching today. For more robust math training,
  replace it with `math_verify`-style answer equivalence.
- `test_crossthink.py` still imports `interleaved_grpo.datasets_loader`, which does not
  exist in this flat repo layout.

## Citation

```bibtex
@misc{akter2025nemotroncrossthinkscalingselflearningmath,
      title={NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning},
      author={Syeda Nahida Akter and Shrimai Prabhumoye and Matvei Novikov and Seungju Han and Ying Lin and Evelina Bakhturina and Eric Nyberg and Yejin Choi and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro},
      year={2025},
      eprint={2504.13941},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.13941},
}
```
