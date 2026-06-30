# Quick side-by-side: what does classic-SFT see as its target vs what does
# SDFT show its teacher as the in-context demo. Both must consume the FULL
# golden_response (Thought + Action + Action Input) so the comparison is not
# confounded by target content. Run before launching phase4 if anything in
# the data pipeline has changed.
#
# Usage:
#   .venv/bin/python verify_training_targets.py [--row 0] [--n 1]

import argparse
import json
from pathlib import Path
from string import Template

from datasets import load_from_disk
from transformers import AutoTokenizer


_REPO = Path(__file__).resolve().parent
TRAIN_DATA_PATH = _REPO / "data/tooluse_data/train_data"
HOLDOUT_PATH = _REPO / "data/tooluse_data/train_subset_holdout_indices.json"

# Byte-identical to train_sdft_lora.py / main.py
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


def filter_holdout(ds):
    with open(HOLDOUT_PATH) as f:
        meta = json.load(f)
    holdout = set(meta["indices"])
    keep = [i for i in range(len(ds)) if i not in holdout]
    return ds.select(keep)


def format_for_classic_sft(example, tokenizer):
    """How train_sft_lora.py builds its training sample."""
    msgs = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant",
         "content": "\n".join(example["golden_response"])},
    ]
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    # SFTTrainer with assistant_only_loss=True trains CE on the assistant
    # span only; we surface that span explicitly here.
    assistant_target = "\n".join(example["golden_response"])
    return full, assistant_target


def format_for_sdft(example):
    """How train_sdft_lora.py builds its teacher_prompt."""
    teacher_prompt = TEACHER_TEMPLATE.substitute(
        orig_content=example["prompt"],
        output_text="\n".join(example["golden_response"]),
    )
    # The student gets only the bare prompt; the teacher gets demo-conditioned.
    student_prompt = example["prompt"]
    return student_prompt, teacher_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--row", type=int, default=0,
                    help="Row index (post-holdout-filter, pre-shuffle).")
    ap.add_argument("--n", type=int, default=1, help="How many rows to print.")
    args = ap.parse_args()

    print(f"Loading tokenizer from {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading dataset from {TRAIN_DATA_PATH} ...")
    ds = load_from_disk(TRAIN_DATA_PATH)
    ds = filter_holdout(ds)
    print(f"  rows after holdout filter: {len(ds)}")

    for offset in range(args.n):
        idx = args.row + offset
        ex = ds[idx]

        sep = "=" * 78
        print(f"\n{sep}\nROW {idx}  api={ex.get('api_name','?')}\n{sep}")
        print("\n[user prompt]")
        print(ex["prompt"])

        print("\n[golden_response — raw list]")
        for i, line in enumerate(ex["golden_response"]):
            print(f"  [{i}] {line!r}")
        joined = "\n".join(ex["golden_response"])
        print(f"\n[golden_response — joined for both trainers] len={len(joined)} chars")
        print(joined)

        full, assistant = format_for_classic_sft(ex, tok)
        print("\n--- CLASSIC SFT (train_sft_lora.py) ---")
        print("Full chat-templated input (assistant_only_loss masks user span):")
        print(full)
        print("Assistant span = TRAINING TARGET (CE on these tokens):")
        print(assistant)

        student_p, teacher_p = format_for_sdft(ex)
        print("\n--- SDFT (train_sdft_lora.py) ---")
        print("Student prompt (bare; student samples y from this on-policy):")
        print(student_p)
        print("\nTeacher prompt (demo-conditioned; teacher computes KL target on y):")
        print(teacher_p)

        ok = assistant.strip() == joined.strip()
        print(f"\n[parity check] classic_sft target == sdft demo content: {ok}")
        if not ok:
            print("!! TARGETS DIFFER — the comparison is confounded.")
        else:
            print("OK — both trainers consume the full golden_response.")


if __name__ == "__main__":
    main()
