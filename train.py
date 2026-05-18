"""Train an interleaved CoT policy with TRL GRPO using LoRA and Nemotron logic."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time
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
    parser.add_argument(
        "--validate-prompts-only",
        action="store_true",
        help="Build and validate the prompt column, print sample prompt tails, then exit before model/trainer loading.",
    )
    parser.add_argument(
        "--prompt-validation-samples",
        type=int,
        default=3,
        help="Number of formatted prompts to inspect during prompt validation.",
    )
    parser.add_argument(
        "--chat-template-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional passthrough for tokenizer.apply_chat_template(enable_thinking=...).",
    )
    parser.add_argument(
        "--assistant-prefill-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append an opening <think> to the assistant prompt when the chat template does not already do it.",
    )
    parser.add_argument(
        "--normalize-prefilled-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a missing opening <think> back to completions before reward parsing when the prompt prefilled it.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated metric sinks for TRL, for example tensorboard or wandb.",
    )
    parser.add_argument(
        "--wandb",
        "-wandb",
        action="store_true",
        help="Also report training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--generation-log-file",
        default=None,
        help="JSONL path for generated completions. Defaults to OUTPUT_DIR/generations.jsonl.",
    )
    parser.add_argument(
        "--generation-log-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prompt text in generation JSONL records when available.",
    )
    parser.add_argument(
        "--disable-generation-logging",
        action="store_true",
        help="Disable JSONL logging of generated completions.",
    )
    parser.add_argument(
        "--profile-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write reward function timing records to JSONL for bottleneck diagnosis.",
    )
    parser.add_argument(
        "--reward-profile-log-file",
        default=None,
        help="JSONL path for reward timing profile records. Defaults to OUTPUT_DIR/reward_profile.jsonl.",
    )
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
        help=(
            "Initial multiplier for the language consistency reward when "
            "--reasoning-lang is used. In legacy gated mode this is the fixed "
            "language reward multiplier."
        ),
    )
    parser.add_argument(
        "--reward-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a smooth one-way reward curriculum based on EMA correctness. "
            "Disable with --no-reward-curriculum to use legacy correctness-gated "
            "format/language rewards."
        ),
    )
    parser.add_argument(
        "--reward-curriculum-ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for batch correctness in the reward curriculum.",
    )
    parser.add_argument(
        "--reward-curriculum-initial-correctness",
        type=float,
        default=0.0,
        help="Initial EMA correctness ratio for reward curriculum mode.",
    )
    parser.add_argument(
        "--reward-curriculum-start-correctness",
        type=float,
        default=0.05,
        help="Best EMA correctness ratio that maps to curriculum progress 0.0.",
    )
    parser.add_argument(
        "--reward-curriculum-end-correctness",
        type=float,
        default=0.60,
        help="Best EMA correctness ratio that maps to curriculum progress 1.0.",
    )
    parser.add_argument(
        "--correctness-reward-weight",
        type=float,
        default=1.0,
        help="Initial correctness reward multiplier for reward curriculum mode.",
    )
    parser.add_argument(
        "--correctness-reward-weight-final",
        type=float,
        default=1.8,
        help="Final correctness reward multiplier for reward curriculum mode.",
    )
    parser.add_argument(
        "--format-reward-weight",
        type=float,
        default=0.5,
        help="Initial XML format reward multiplier for reward curriculum mode.",
    )
    parser.add_argument(
        "--format-reward-weight-final",
        type=float,
        default=0.1,
        help="Final XML format reward multiplier for reward curriculum mode.",
    )
    parser.add_argument(
        "--language-reward-weight-final",
        type=float,
        default=0.1,
        help="Final language consistency reward multiplier for reward curriculum mode.",
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


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _column_item(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, (str, bytes, dict)):
        return values
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _stable_hash(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def assistant_generation_tail(prompt: str) -> str:
    """Return only the generated-assistant prefix of a formatted chat prompt."""
    for marker in assistant_generation_markers():
        index = prompt.rfind(marker)
        if index >= 0:
            return prompt[index + len(marker):]
    return prompt[-256:]


def assistant_generation_markers() -> tuple[str, ...]:
    """Known assistant generation markers used by common chat templates."""
    return (
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<assistant>",
        "assistant\n",
    )


def has_assistant_generation_marker(prompt: Any) -> bool:
    text = "" if prompt is None else str(prompt)
    return any(marker in text for marker in assistant_generation_markers())


def prompt_prefills_open_think(prompt: Any) -> bool:
    """Return whether the prompt leaves an unmatched opening <think> for generation."""
    text = "" if prompt is None else str(prompt)
    assistant_tail = assistant_generation_tail(text)
    stripped = assistant_tail.lstrip()
    return stripped.startswith("<think>") and assistant_tail.count("<think>") > assistant_tail.count("</think>")


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool | None = None,
    assistant_prefill_think: bool = True,
) -> str:
    """Apply the tokenizer chat template with optional Qwen thinking-mode handling."""
    if enable_thinking is False and assistant_prefill_think:
        print(
            "Warning: --no-chat-template-enable-thinking asks the tokenizer for "
            "non-thinking mode, but --assistant-prefill-think will append a new "
            "open <think>. For Qwen3 reasoning runs, leave "
            "--chat-template-enable-thinking unset or set it true."
        )
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if assistant_prefill_think and not prompt_prefills_open_think(prompt):
        return prompt + "<think>\n"
    return prompt


def normalize_prefilled_think_completion(prompt: Any, completion: Any) -> str:
    """Reconstruct a prompt-prefilled opening <think> before reward parsing."""
    completion_text = _completion_text(completion)
    if not prompt_prefills_open_think(prompt):
        return completion_text
    if completion_text.lstrip().startswith("<think>"):
        return completion_text
    return "<think>" + completion_text


def normalize_prefilled_think_completions(
    prompts: list[Any],
    completions: list[Any],
    enabled: bool = True,
) -> list[str]:
    """Normalize a batch of completions for templates/prompts that prefilled <think>."""
    if not enabled:
        return [_completion_text(completion) for completion in completions]
    if len(prompts) == len(completions):
        return [
            normalize_prefilled_think_completion(prompt, completion)
            for prompt, completion in zip(prompts, completions)
        ]
    if len(prompts) == 1:
        return [
            normalize_prefilled_think_completion(prompts[0], completion)
            for completion in completions
        ]
    return [_completion_text(completion) for completion in completions]


def validate_prompt_column(
    dataset: Any,
    sample_count: int = 3,
    require_prefilled_think: bool = True,
) -> None:
    """Print and validate prompt-column alignment with the assistant generation tail."""
    total = len(dataset)
    count = min(max(sample_count, 0), total)
    if count == 0:
        print("Prompt validation skipped: dataset is empty.")
        return

    marker_count = 0
    prefill_count = 0
    for index in range(count):
        prompt = dataset[index]["prompt"]
        has_marker = has_assistant_generation_marker(prompt)
        prefills_think = prompt_prefills_open_think(prompt)
        marker_count += int(has_marker)
        prefill_count += int(prefills_think)
        tail = assistant_generation_tail(str(prompt))
        print(
            f"Prompt validation sample {index}: "
            f"has_assistant_marker={has_marker} "
            f"prefills_open_think={prefills_think} "
            f"assistant_tail={tail[-200:]!r}"
        )
        if "<think>\n\n</think>" in tail and prefills_think:
            print(
                "Warning: prompt tail contains Qwen3 non-thinking prefill "
                "`<think>\\n\\n</think>` plus an open <think>. For reasoning runs, "
                "prefer leaving --chat-template-enable-thinking unset or true."
            )

    print(
        "Prompt validation summary: "
        f"checked={count} assistant_marker={marker_count}/{count} "
        f"prefilled_open_think={prefill_count}/{count}"
    )
    if marker_count != count:
        raise ValueError(
            "Formatted prompt column is missing a recognized assistant generation marker."
        )
    if require_prefilled_think and prefill_count != count:
        raise ValueError(
            "Formatted prompt column does not leave an unmatched assistant <think> "
            "prefix. For Qwen3-style reasoning, keep --assistant-prefill-think enabled "
            "and avoid --no-chat-template-enable-thinking."
        )


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _smoothstep(progress: float) -> float:
    progress = _clamp(progress, 0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


class RewardCurriculum:
    """Smooth one-way reward weights based on best EMA correctness."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.ema_decay = _clamp(args.reward_curriculum_ema_decay, 0.0, 0.999999)
        self.start_correctness = args.reward_curriculum_start_correctness
        self.end_correctness = args.reward_curriculum_end_correctness
        self.correctness_weight_start = args.correctness_reward_weight
        self.correctness_weight_final = args.correctness_reward_weight_final
        self.format_weight_start = args.format_reward_weight
        self.format_weight_final = args.format_reward_weight_final
        self.language_weight_start = args.language_reward_weight
        self.language_weight_final = args.language_reward_weight_final
        self.ema_correctness = _clamp(
            args.reward_curriculum_initial_correctness,
            0.0,
            1.0,
        )
        self.best_ema_correctness = self.ema_correctness
        self.last_batch_correctness = 0.0
        self.update_count = 0

    def update(self, correctness_values: list[float]) -> None:
        numeric = [
            float(value)
            for value in correctness_values
            if isinstance(value, (int, float))
        ]
        batch_correctness = sum(numeric) / max(len(numeric), 1)
        self.last_batch_correctness = batch_correctness
        self.ema_correctness = (
            self.ema_decay * self.ema_correctness
            + (1.0 - self.ema_decay) * batch_correctness
        )
        self.best_ema_correctness = max(
            self.best_ema_correctness,
            self.ema_correctness,
        )
        self.update_count += 1

    def progress(self) -> float:
        span = self.end_correctness - self.start_correctness
        if span <= 0.0:
            raw_progress = 1.0 if self.best_ema_correctness >= self.end_correctness else 0.0
        else:
            raw_progress = (
                self.best_ema_correctness - self.start_correctness
            ) / span
        return _smoothstep(raw_progress)

    def weights(self) -> dict[str, float]:
        progress = self.progress()
        return {
            "correctness_weight": _lerp(
                self.correctness_weight_start,
                self.correctness_weight_final,
                progress,
            ),
            "format_weight": _lerp(
                self.format_weight_start,
                self.format_weight_final,
                progress,
            ),
            "language_weight": _lerp(
                self.language_weight_start,
                self.language_weight_final,
                progress,
            ),
        }

    def snapshot(self) -> dict[str, float | int]:
        progress = self.progress()
        return {
            "update_count": self.update_count,
            "last_batch_correctness": self.last_batch_correctness,
            "ema_correctness": self.ema_correctness,
            "best_ema_correctness": self.best_ema_correctness,
            "start_correctness": self.start_correctness,
            "end_correctness": self.end_correctness,
            "progress": progress,
            **self.weights(),
        }

    def weighted_correctness_reward(self, completions, answer, **kwargs) -> list[float]:
        rewards = correctness_reward(completions, answer=answer, **kwargs)
        self.update(rewards)
        weight = self.weights()["correctness_weight"]
        return [weight * reward for reward in rewards]

    def weighted_xml_format_reward(self, completions, **kwargs) -> list[float]:
        rewards = xml_format_reward(completions, **kwargs)
        weight = self.weights()["format_weight"]
        return [weight * reward for reward in rewards]

    def weighted_language_consistency_reward(self, completions, **kwargs) -> list[float]:
        rewards = language_consistency_reward(completions, **kwargs)
        weight = self.weights()["language_weight"]
        return [weight * reward for reward in rewards]


def with_prefilled_think_normalization(reward_func, enabled: bool = True):
    """Wrap a reward function so prompt-prefilled <think> completions parse correctly."""

    @functools.wraps(reward_func)
    def wrapped(completions, *args, **kwargs):
        prompt_values = kwargs.get("prompts", kwargs.get("prompt"))
        completion_list = _as_list(completions)
        if prompt_values is None:
            normalized = [_completion_text(completion) for completion in completion_list]
        else:
            normalized = normalize_prefilled_think_completions(
                _as_list(prompt_values),
                completion_list,
                enabled=enabled,
            )
        return reward_func(normalized, *args, **kwargs)

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
    has_language_reward = args.reasoning_lang or dataset_has_reasoning_lang(dataset)
    curriculum = RewardCurriculum(args) if args.reward_curriculum else None

    if curriculum is not None:
        reward_funcs = [
            with_prefilled_think_normalization(
                curriculum.weighted_correctness_reward,
                enabled=args.normalize_prefilled_think,
            ),
            with_prefilled_think_normalization(
                curriculum.weighted_xml_format_reward,
                enabled=args.normalize_prefilled_think,
            ),
        ]
        if has_language_reward:
            reward_funcs.append(
                with_prefilled_think_normalization(
                    curriculum.weighted_language_consistency_reward,
                    enabled=args.normalize_prefilled_think,
                )
            )
    else:
        reward_funcs = [
            with_prefilled_think_normalization(
                correctness_reward,
                enabled=args.normalize_prefilled_think,
            ),
            with_prefilled_think_normalization(
                correctness_gated_reward(xml_format_reward),
                enabled=args.normalize_prefilled_think,
            ),
        ]
        if has_language_reward:
            reward_funcs.append(
                with_prefilled_think_normalization(
                    correctness_gated_reward(
                        weighted_reward(
                            language_consistency_reward,
                            args.language_reward_weight,
                        )
                    ),
                    enabled=args.normalize_prefilled_think,
                )
            )

    if not args.disable_generation_logging:
        reward_funcs.append(
            GenerationJsonlLogger(args, curriculum=curriculum).generation_jsonl_logger
        )
    if args.profile_rewards:
        profiler = RewardTimingProfiler(args)
        reward_funcs = [profiler.wrap(reward_func) for reward_func in reward_funcs]
    return reward_funcs


def generation_log_path(args: argparse.Namespace) -> Path:
    return Path(args.generation_log_file or Path(args.output_dir) / "generations.jsonl")


def reward_profile_log_path(args: argparse.Namespace) -> Path:
    return Path(args.reward_profile_log_file or Path(args.output_dir) / "reward_profile.jsonl")


def resolve_report_to(args: argparse.Namespace) -> str | list[str]:
    """Resolve metric reporting targets from --report-to and --wandb."""
    if args.report_to in {None, "", "none"}:
        report_targets: list[str] = []
    else:
        report_targets = [
            target.strip()
            for target in str(args.report_to).split(",")
            if target.strip()
        ]

    if args.wandb and "wandb" not in report_targets:
        report_targets.append("wandb")
    return report_targets if report_targets else "none"


def supported_init_kwargs(config_cls: Any) -> set[str] | None:
    """Return supported __init__ kwargs, or None when arbitrary kwargs are accepted."""
    try:
        signature = inspect.signature(config_cls.__init__)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        supported: set[str] = set()
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return None
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                supported.add(name)
        if supported:
            return supported

    if is_dataclass(config_cls):
        return {field.name for field in fields(config_cls)}
    return None


def filter_supported_init_kwargs(
    config_cls: Any,
    kwargs: dict[str, Any],
    object_name: str,
) -> dict[str, Any]:
    """Drop kwargs unsupported by the installed dependency version."""
    supported = supported_init_kwargs(config_cls)
    if supported is None:
        return kwargs

    filtered = {name: value for name, value in kwargs.items() if name in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(
            f"{object_name} does not support these arguments; ignoring them: "
            f"{', '.join(dropped)}"
        )
    return filtered


def build_grpo_config(config_cls: Any, args: argparse.Namespace) -> Any:
    """Build GRPOConfig across TRL versions with different constructor signatures."""
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "beta": args.beta,
        "remove_unused_columns": False,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": args.bf16,
        "seed": args.seed,
        "report_to": resolve_report_to(args),
    }

    supported = supported_init_kwargs(config_cls)
    if (
        supported is not None
        and "max_completion_length" not in supported
        and "generation_kwargs" in supported
    ):
        kwargs["generation_kwargs"] = {
            "max_new_tokens": args.max_completion_length,
            "temperature": args.temperature,
        }

    return config_cls(
        **filter_supported_init_kwargs(config_cls, kwargs, "GRPOConfig")
    )


class RewardTimingProfiler:
    """Write compact timing records for reward function calls."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.path = reward_profile_log_path(args)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.print_every = max(args.logging_steps, 1)
        self._stats: dict[str, dict[str, float]] = {}

    def wrap(self, reward_func):
        @functools.wraps(reward_func)
        def wrapped(completions, *args, **kwargs):
            completion_count = len(_as_list(completions))
            start = time.perf_counter()
            rewards = reward_func(completions, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(reward_func.__name__, elapsed_ms, completion_count)
            return rewards

        return wrapped

    def record(self, reward_name: str, elapsed_ms: float, completion_count: int) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        record = {
            "created_at": time.time(),
            "process_rank": rank,
            "reward_name": reward_name,
            "elapsed_ms": elapsed_ms,
            "completion_count": completion_count,
            "ms_per_completion": elapsed_ms / max(completion_count, 1),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

        stats = self._stats.setdefault(
            reward_name,
            {"calls": 0.0, "elapsed_ms": 0.0, "completion_count": 0.0},
        )
        stats["calls"] += 1.0
        stats["elapsed_ms"] += elapsed_ms
        stats["completion_count"] += float(completion_count)
        if int(stats["calls"]) % self.print_every == 0:
            avg_batch_ms = stats["elapsed_ms"] / stats["calls"]
            avg_item_ms = stats["elapsed_ms"] / max(stats["completion_count"], 1.0)
            print(
                f"[reward-profile rank={rank}] {reward_name}: "
                f"avg_batch_ms={avg_batch_ms:.3f} "
                f"avg_item_ms={avg_item_ms:.3f} "
                f"calls={int(stats['calls'])}"
            )


class GenerationJsonlLogger:
    """Write one JSONL record per generated completion without changing rewards."""

    def __init__(
        self,
        args: argparse.Namespace,
        curriculum: RewardCurriculum | None = None,
    ) -> None:
        self.path = generation_log_path(args)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_prompts = args.generation_log_prompts
        self.language_reward_weight = args.language_reward_weight
        self.normalize_prefilled_think = args.normalize_prefilled_think
        self.curriculum = curriculum
        self._next_batch_id = 0

    def generation_jsonl_logger(
        self,
        completions,
        answer=None,
        prompt=None,
        prompts=None,
        reasoning_lang=None,
        reasoning_language=None,
        language=None,
        task_type=None,
        **kwargs,
    ) -> list[float]:
        completion_list = _as_list(completions)
        answer_list = _as_list(answer) if answer is not None else None
        prompt_values = prompts if prompts is not None else prompt
        prompt_list = _as_list(prompt_values) if prompt_values is not None else []
        normalized_completion_list = (
            normalize_prefilled_think_completions(
                prompt_list,
                completion_list,
                enabled=self.normalize_prefilled_think,
            )
            if prompt_values is not None
            else [_completion_text(completion) for completion in completion_list]
        )

        correctness = (
            correctness_reward(normalized_completion_list, answer=answer_list, **kwargs)
            if answer_list is not None
            else [None] * len(normalized_completion_list)
        )
        format_rewards = xml_format_reward(normalized_completion_list, **kwargs)
        gated_format = [
            reward if correct and correct > 0.0 else 0.0
            for reward, correct in zip(format_rewards, correctness)
        ]

        language_values = reasoning_lang
        if language_values is None:
            language_values = reasoning_language
        if language_values is None:
            language_values = language
        language_rewards: list[float | None]
        gated_language: list[float | None]
        if language_values is not None:
            language_rewards = language_consistency_reward(
                normalized_completion_list,
                reasoning_lang=language_values,
                **kwargs,
            )
            gated_language = [
                self.language_reward_weight * reward if correct and correct > 0.0 else 0.0
                for reward, correct in zip(language_rewards, correctness)
            ]
        else:
            language_rewards = [None] * len(completion_list)
            gated_language = [None] * len(completion_list)

        curriculum_snapshot = (
            self.curriculum.snapshot() if self.curriculum is not None else None
        )
        if curriculum_snapshot is not None:
            correctness_weight = float(curriculum_snapshot["correctness_weight"])
            format_weight = float(curriculum_snapshot["format_weight"])
            language_weight = float(curriculum_snapshot["language_weight"])
            weighted_correctness = [
                correctness_weight * reward
                if isinstance(reward, (int, float))
                else None
                for reward in correctness
            ]
            weighted_format = [
                format_weight * reward
                if isinstance(reward, (int, float))
                else None
                for reward in format_rewards
            ]
            weighted_language = [
                language_weight * reward
                if isinstance(reward, (int, float))
                else None
                for reward in language_rewards
            ]
        else:
            weighted_correctness = correctness
            weighted_format = gated_format
            weighted_language = gated_language

        records = []
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        created_at = time.time()
        for index, completion in enumerate(completion_list):
            normalized_completion = _column_item(normalized_completion_list, index)
            correct = _column_item(correctness, index)
            fmt = _column_item(format_rewards, index)
            gated_fmt = _column_item(gated_format, index)
            lang = _column_item(language_rewards, index)
            gated_lang = _column_item(gated_language, index)
            weighted_correct = _column_item(weighted_correctness, index)
            weighted_fmt = _column_item(weighted_format, index)
            weighted_lang = _column_item(weighted_language, index)
            weighted_reward = 0.0
            for value in (weighted_correct, weighted_fmt, weighted_lang):
                if isinstance(value, (int, float)):
                    weighted_reward += float(value)

            record = {
                "batch_id": batch_id,
                "generation_index": index,
                "created_at": created_at,
                "process_rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                "completion": _jsonable(completion),
                "answer": _jsonable(_column_item(answer, index)),
                "task_type": _jsonable(_column_item(task_type, index)),
                "reasoning_lang": _jsonable(_column_item(language_values, index)),
                "rewards": {
                    "correctness_reward": correct,
                    "xml_format_reward": fmt,
                    "correctness_gated_xml_format_reward": gated_fmt,
                    "language_consistency_reward": lang,
                    "correctness_gated_weighted_language_consistency_reward": gated_lang,
                    "weighted_correctness_reward": weighted_correct,
                    "weighted_xml_format_reward": weighted_fmt,
                    "weighted_language_consistency_reward": weighted_lang,
                },
                "weighted_reward": weighted_reward,
            }
            if curriculum_snapshot is not None:
                record["reward_curriculum"] = _jsonable(curriculum_snapshot)
            if normalized_completion != _completion_text(completion):
                record["normalized_completion"] = _jsonable(normalized_completion)
            prompt_item = _column_item(prompt_values, index)
            if self.include_prompts:
                record["prompt"] = _jsonable(prompt_item)
            else:
                record["prompt_hash"] = _stable_hash(prompt_item)
                record["prompt_char_len"] = len("" if prompt_item is None else str(prompt_item))
            records.append(record)

        self._append_jsonl(records)
        return [0.0] * len(completion_list)

    def _append_jsonl(self, records: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass

            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def main():
    parser = build_parser()
    args = parser.parse_args()

    from transformers import AutoTokenizer

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
        prompt = apply_chat_template_text(
            tokenizer,
            messages,
            enable_thinking=args.chat_template_enable_thinking,
            assistant_prefill_think=args.assistant_prefill_think,
        )
        formatted = {"prompt": prompt}
        if use_reasoning_lang:
            formatted["reasoning_lang"] = reasoning_lang
        return formatted

    dataset = raw_dataset.map(format_reward_dataset)
    validate_prompt_column(
        dataset,
        sample_count=args.prompt_validation_samples,
        require_prefilled_think=args.assistant_prefill_think,
    )
    if args.validate_prompts_only:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

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
    training_args = build_grpo_config(GRPOConfig, args)

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
