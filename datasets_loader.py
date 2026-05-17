"""Dataset loading helpers for GRPO interleaved reasoning training."""

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datasets import Dataset


DatasetName = Literal["gsm8k", "math", "nemotron-crossthink"]
CROSSTHINK_MATH_DATA_FILE = (
    "hf://datasets/nvidia/Nemotron-CrossThink/Data/Nemotron-CrossThink-Math.jsonl"
)

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
    "zh-hans": "Simplified Chinese",
    "yue": "Cantonese",
}

INTERLEAVED_SYSTEM_PROMPT = (
    "You are an interleaved reasoning agent. For every step: "
    "Start with exactly one <think> block and close every tag you open. "
    "Never output </think> unless it closes a matching <think> block. "
    "1. Use <think> to plan. "
    "2. Perform an action, either a Math <answer>...</answer> or a <tool_call>...</tool_call>. "
    "3. Use <think> to reflect on the result. "
    "Be concise; reach the first action as quickly as possible."
)


def build_interleaved_messages(
    question: str,
    task_type: str = "math",
    reasoning_lang: str | None = None,
    reasoning_language: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages that guide the policy toward interleaved reasoning."""
    target_language = reasoning_lang or reasoning_language
    if target_language:
        language = _language_name(target_language)
        user_content = (
            f"Solve this step-by-step. Write all private reasoning inside "
            f"<think>...</think> in {language}. Put only the final math result "
            f"in <answer>...</answer>: {question.strip()}"
        )
    else:
        user_content = f"Solve this step-by-step. Put the final math result in <answer>...</answer>: {question.strip()}"
    return [
        {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(
        str(language).strip().lower(),
        str(language).strip() or "English",
    )


def _extract_gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


def load_training_dataset(
    dataset_name: DatasetName = "gsm8k",
    split: str = "train",
    subset: str | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load and normalize a math dataset for TRL GRPO."""
    from datasets import load_dataset

    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", subset or "main", split=split)

        def process_gsm8k(batch: dict[str, list[str]]) -> dict[str, list[str]]:
            return {
                "question": [question.strip() for question in batch["question"]],
                "answer": [_extract_gsm8k_answer(answer) for answer in batch["answer"]],
                "task_type": ["math"] * len(batch["question"]),
            }

        dataset = dataset.map(
            process_gsm8k, batched=True, remove_columns=dataset.column_names
        )

    elif dataset_name == "math":
        dataset = load_dataset(
            "hendrycks/competition_math", subset or "all", split=split
        )

        def process_math(batch: dict[str, list[str]]) -> dict[str, list[str]]:
            return {
                "question": [problem.strip() for problem in batch["problem"]],
                "answer": [solution.strip() for solution in batch["solution"]],
                "task_type": ["math"] * len(batch["problem"]),
            }

        dataset = dataset.map(
            process_math, batched=True, remove_columns=dataset.column_names
        )

    elif dataset_name == "nemotron-crossthink":
        if split not in {"train", "train_math"}:
            raise ValueError(
                "Nemotron-CrossThink math data is available locally as the "
                f"train split; got split={split!r}."
            )

        # Loading the default dataset config can infer a too-narrow schema from
        # the QA split before reading math rows. Load the math JSONL directly.
        dataset = load_dataset(
            "json",
            data_files=CROSSTHINK_MATH_DATA_FILE,
            split="train",
        )

        def process_nemotron_crossthink(
            batch: dict[str, list[dict]],
        ) -> dict[str, list[str]]:
            # Field structure verified via exploration:
            # meta_data['question'], reward_model['ground_truth']
            questions = [item.get("question", "") for item in batch["meta_data"]]
            answers = [item.get("ground_truth", "") for item in batch["reward_model"]]
            return {
                "question": questions,
                "answer": answers,
                "task_type": ["math"] * len(questions),
            }

        dataset = dataset.map(
            process_nemotron_crossthink,
            batched=True,
            remove_columns=dataset.column_names,
        )

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(max_samples, len(dataset)))
        )

    return dataset


def dataset_from_args(args: Namespace) -> Dataset:
    """Build a dataset from CLI args."""
    return load_training_dataset(
        dataset_name=args.dataset,
        split=args.dataset_split,
        subset=args.dataset_subset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
