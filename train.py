"""Train an interleaved CoT policy with TRL GRPO using LoRA and Nemotron logic."""

from __future__ import annotations

import argparse
import functools
import os
import re
from typing import Any

from datasets_loader import dataset_from_args, build_interleaved_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "nemotron-crossthink"],
        default="nemotron-crossthink",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="./nemo_rl_output")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--reasoning-lang",
        "--reasoning_lang",
        "--reansoning_lang",
        dest="reasoning_lang",
        default=None,
        help="Fallback language for private <think> reasoning, for example yue, zh, or en.",
    )
    parser.add_argument(
        "--language-reward-weight",
        type=float,
        default=0.2,
        help="Multiplier for the language consistency reward when --reasoning-lang is used.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser


# --- Reward Functions ---


def xml_format_reward(completions, **kwargs) -> list[float]:
    """Reward for using <think> and <answer> tags correctly."""
    rewards = []
    for content in completions:
        score = 0.0
        if "<think>" in content and "</think>" in content:
            score += 0.2
        if "<answer>" in content and "</answer>" in content:
            score += 0.3
        # Check interleaving or multiple think blocks if desired
        if content.count("<think>") >= 1:
            score += 0.1
        rewards.append(score)
    return rewards


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """Reward for matching the ground truth answer."""
    rewards = []
    for content, ground_truth in zip(completions, answer):
        # Extract content from <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Simple string match for now, could use math_verify for complex math
            if extracted == ground_truth.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def _indexed_item(values: Any, index: int) -> Any | None:
    if isinstance(values, (str, bytes, dict)):
        return None
    try:
        if index < len(values):
            return values[index]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def _get_language_targets(
    count: int,
    reasoning_lang: Any = None,
    reasoning_language: Any = None,
    language: Any = None,
) -> list[str | None]:
    values = reasoning_lang
    if values is None:
        values = reasoning_language
    if values is None:
        values = language
    if values is None:
        return [None] * count
    if isinstance(values, str):
        return [values] * count

    targets: list[str | None] = []
    for index in range(count):
        item = _indexed_item(values, index)
        targets.append(str(item) if item is not None else None)
    return targets


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def _reasoning_text_for_language(completion: Any) -> str:
    content = _completion_text(completion)
    think_blocks = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_blocks:
        return " ".join(block.strip() for block in think_blocks if block.strip())
    if "</think>" in content:
        return content.split("</think>", 1)[0].replace("<think>", " ").strip()
    return ""


def _cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _latin_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text))


def _looks_cantonese(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    try:
        from cantofilter import judge as yue_judge

        judgement = yue_judge(stripped)
        if judgement in {"cantonese", "mixed"}:
            return True
        if judgement == "neutral" and _cjk_count(stripped) > 0:
            return True
    except Exception:
        pass

    cantonese_markers = (
        "嘅",
        "咗",
        "佢",
        "哋",
        "唔",
        "喺",
        "呢",
        "咁",
        "啲",
        "嚟",
        "晒",
        "冇",
        "嗰",
        "啦",
        "呀",
        "㗎",
        "咩",
    )
    cjk = _cjk_count(stripped)
    return cjk > 0 and (
        any(marker in stripped for marker in cantonese_markers) or cjk >= 12
    )


def _matches_language(text: str, language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().lower()
    if normalized == "":
        return False
    if normalized == "yue":
        return _looks_cantonese(text)
    if normalized in {"zh", "zh-hant", "zh-hans", "cn"}:
        return _cjk_count(text) >= 4
    if normalized == "en":
        return _latin_word_count(text) >= 4 and _cjk_count(text) == 0
    return True


def language_consistency_reward(
    completions,
    reasoning_lang=None,
    reasoning_language=None,
    language=None,
    **kwargs,
) -> list[float]:
    """Reward <think> text that matches the requested reasoning language."""
    targets = _get_language_targets(
        len(completions),
        reasoning_lang=reasoning_lang,
        reasoning_language=reasoning_language,
        language=language,
    )
    rewards = []
    for completion, target_language in zip(completions, targets):
        reasoning_text = _reasoning_text_for_language(completion)
        if not reasoning_text or target_language is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if _matches_language(reasoning_text, target_language) else 0.0)
    return rewards


def weighted_reward(reward_func, weight: float):
    """Return a reward function scaled for TRL versions without reward_weights."""
    if weight == 1.0:
        return reward_func

    @functools.wraps(reward_func)
    def wrapped(*args, **kwargs):
        return [weight * reward for reward in reward_func(*args, **kwargs)]

    return wrapped


def correctness_gated_reward(reward_func):
    """Return a reward function that only scores when the final answer is correct."""

    @functools.wraps(reward_func)
    def wrapped(completions, answer, **kwargs):
        gates = correctness_reward(completions, answer=answer, **kwargs)
        rewards = reward_func(completions, answer=answer, **kwargs)
        return [
            reward if gate > 0.0 else 0.0
            for reward, gate in zip(rewards, gates)
        ]

    wrapped.__name__ = f"correctness_gated_{reward_func.__name__}"
    return wrapped


def dataset_has_reasoning_lang(dataset: Any) -> bool:
    names = getattr(dataset, "column_names", None)
    if names is None and isinstance(dataset, dict):
        names = dataset.get("column_names", dataset.keys())
    return names is not None and any(
        name in names for name in ("reasoning_lang", "reasoning_language", "language")
    )


def resolve_reasoning_lang(example: dict[str, Any], default: str | None) -> str | None:
    for name in ("reasoning_lang", "reasoning_language", "language"):
        value = example.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def select_reward_funcs(args: argparse.Namespace, dataset: Any) -> list[Any]:
    reward_funcs = [correctness_reward, correctness_gated_reward(xml_format_reward)]
    if args.reasoning_lang or dataset_has_reasoning_lang(dataset):
        reward_funcs.append(
            correctness_gated_reward(
                weighted_reward(
                    language_consistency_reward,
                    args.language_reward_weight,
                )
            )
        )
    return reward_funcs


def main():
    parser = build_parser()
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # 1. Load Dataset
    raw_dataset = dataset_from_args(args)
    use_reasoning_lang = bool(args.reasoning_lang) or dataset_has_reasoning_lang(
        raw_dataset
    )

    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-process dataset to include 'prompt' field using chat template
    def format_reward_dataset(example):
        reasoning_lang = resolve_reasoning_lang(example, args.reasoning_lang)
        messages = build_interleaved_messages(
            example["question"],
            task_type=example.get("task_type", "math"),
            reasoning_lang=reasoning_lang,
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted = {"prompt": prompt}
        if use_reasoning_lang:
            formatted["reasoning_lang"] = reasoning_lang
        return formatted

    dataset = raw_dataset.map(format_reward_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # 3. Training Arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        seed=args.seed,
        report_to="none",  # Change to tensorboard or wandb as needed
    )

    # 4. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=select_reward_funcs(args, dataset),
        args=training_args,
        train_dataset=dataset,
    )

    # 5. Train
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
