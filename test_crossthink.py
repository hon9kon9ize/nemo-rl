import os
from datasets import load_dataset
from interleaved_grpo.datasets_loader import build_interleaved_messages


def _normalize_nemotron_crossthink(example):
    """Normalize a row from nvidia/Nemotron-CrossThink dataset."""
    # The dataset has meta_data['question'] and reward_model['ground_truth']
    question = example["meta_data"].get("question", "")
    answer = example["reward_model"].get("ground_truth", "")

    # Nemotron-CrossThink often has a 'prompt' field which is a list of messages.
    # But for our training script, we want to construct our own interleaved prompt.
    messages = build_interleaved_messages(question)

    return {
        "prompt": messages,
        "answer": answer,
    }


def create_crossthink_dataset(split="train_math", max_samples=None):
    """Load and normalize Nemotron-CrossThink."""
    ds = load_dataset("nvidia/Nemotron-CrossThink", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    ds = ds.map(_normalize_nemotron_crossthink)
    return ds


if __name__ == "__main__":
    # Test loading
    ds = create_crossthink_dataset(max_samples=5)
    print(f"Loaded {len(ds)} samples")
    print("First sample prompt:", ds[0]["prompt"])
    print("First sample answer:", ds[0]["answer"])
