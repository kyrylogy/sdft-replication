# Classic OFFLINE SFT-LoRA trainer. No teacher, no rollouts, no KL — plain
# token-level cross-entropy on golden_response with response-only loss masking.
#
# This is the canonical SFT baseline the SDFT paper (and the Khamis et al.
# reproduction) compare against. It is DELIBERATELY different from the
# "online SFT" path in train_sdft_lora.py (--generate_from_teacher), which
# uses the teacher to roll out completions every step. That one is an
# on-policy-isolation ablation, NOT the classic SFT baseline.
#
# Memory: only the student + LoRA + grads + optimizer state are resident.
# No teacher model, no vLLM. 3B and 7B both fit easily on a single 40GB A100.
#
# CLI surface mirrors train_sdft_lora.py for orchestrator parity, minus the
# trainer-specific knobs (--use_vllm, --teacher_adapter_ema, etc).

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent
HOLDOUT_PATH = _REPO / "data/tooluse_data/train_subset_holdout_indices.json"
TRAIN_DATA_PATH = _REPO / "data/tooluse_data/train_data"


# ---------------------------------------------------------------------------
# CLI — mirrors train_sdft_lora.py for orchestrator parity
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Classic offline SFT-LoRA trainer (no teacher)")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_holdout_filter", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=1024,
                   help="Soft hint; SFTConfig.max_length actually caps token count below.")
    p.add_argument("--max_completion_length", type=int, default=1024,
                   help="Soft hint; same as above. Sequence is prompt+completion, capped by max_length.")
    p.add_argument("--report_to", type=str, default="none",
                   choices=["none", "wandb", "tensorboard"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="sdft-replication")
    # Checkpointing / resume — same surface as train_sdft_lora.py.
    p.add_argument("--save_steps", type=int, default=1_000_000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Holdout filter (byte-identical contract to train_sdft_lora.py)
# ---------------------------------------------------------------------------
def filter_holdout(raw_dataset, no_holdout_filter: bool):
    if no_holdout_filter:
        print(f"[holdout] --no_holdout_filter set; keeping all {len(raw_dataset)} rows")
        return raw_dataset
    import json
    with open(HOLDOUT_PATH, "r") as f:
        meta = json.load(f)
    holdout = set(meta["indices"])
    assert meta["source_rows"] == len(raw_dataset), (
        f"holdout file expects {meta['source_rows']} rows but dataset has {len(raw_dataset)}"
    )
    keep_idx = [i for i in range(len(raw_dataset)) if i not in holdout]
    filtered = raw_dataset.select(keep_idx)
    print(f"[holdout] dropped {len(holdout)} indices; {len(raw_dataset)} -> {len(filtered)} rows")
    return filtered


# ---------------------------------------------------------------------------
# Dataset: format as chat template (user=prompt, assistant=golden_response)
# ---------------------------------------------------------------------------
def load_and_format_dataset(tokenizer, seed: int, no_holdout_filter: bool, max_samples):
    raw_dataset = load_from_disk(TRAIN_DATA_PATH)
    print(f"[dataset] loaded {len(raw_dataset)} raw rows from {TRAIN_DATA_PATH}")
    raw_dataset = filter_holdout(raw_dataset, no_holdout_filter)

    def format_example(example):
        # Pre-templatize as a single string. DataCollatorForCompletionOnlyLM
        # later masks everything before the assistant marker so the loss
        # is computed only on the assistant tokens (= golden_response).
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant",
                 "content": "\n".join(example["golden_response"])},
            ],
            tokenize=False,
        )
        return {"text": text}

    formatted = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    formatted = formatted.shuffle(seed=seed)

    if max_samples is not None:
        n = min(max_samples, len(formatted))
        formatted = formatted.select(range(n))
        print(f"[dataset] truncated to --max_samples={max_samples}; final size {len(formatted)}")

    return formatted


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(model_name: str, bf16: bool):
    dtype = torch.bfloat16 if bf16 else torch.float32
    print(f"[model] loading {model_name} ({dtype})")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, lora_r: int, lora_alpha: int):
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[lora] trainable {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return peft_model


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print(f"[wandb] project={args.wandb_project} run_name={args.run_name or 'auto'}")

    model, tokenizer = build_model_and_tokenizer(args.model_name, args.bf16)
    model_peft = apply_lora(model, args.lora_r, args.lora_alpha)
    model_peft.enable_input_require_grads()

    dataset = load_and_format_dataset(
        tokenizer=tokenizer,
        seed=args.seed,
        no_holdout_filter=args.no_holdout_filter,
        max_samples=args.max_samples,
    )

    # Total sequence cap. prompt+completion lengths inform but don't directly
    # set this — SFTTrainer truncates at max_length.
    max_length = args.max_prompt_length + args.max_completion_length

    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps" if args.save_steps < 1_000_000 else "no",
        report_to=args.report_to,
        run_name=args.run_name or os.path.basename(args.output_dir.rstrip("/")),
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=args.seed,
        # Sequence handling.
        max_length=max_length,
        packing=False,
        dataset_text_field="text",
        # NOTE: NOT using assistant_only_loss=True — that requires the
        # tokenizer's chat template to contain {% generation %} markers,
        # which Qwen2.5's default template doesn't. Response-only loss is
        # handled by DataCollatorForCompletionOnlyLM below instead.
    )

    # Response-only loss masking via the canonical TRL collator. Uses token
    # IDs (not the raw string) so it matches deterministically regardless of
    # tokenization edge cases at the boundary.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False,
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model_peft,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    resume_arg = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
            if ckpts:
                resume_arg = str(ckpts[-1])
                print(f"[resume] auto-detected latest checkpoint: {resume_arg}")
        else:
            resume_arg = args.resume_from_checkpoint
            print(f"[resume] using checkpoint: {resume_arg}")
    trainer.train(resume_from_checkpoint=resume_arg)

    if trainer.accelerator.is_main_process:
        adapter_dir = os.path.join(args.output_dir, "lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model_peft.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[save] LoRA adapter written to {adapter_dir}")


if __name__ == "__main__":
    main()
