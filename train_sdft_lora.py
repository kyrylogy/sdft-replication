# SDFT (Self-Distillation Fine-Tuning) LoRA trainer — laptop / Apple-Silicon (MPS) edition.
# The teacher is the SAME base model conditioned in-context on a golden demonstration
# (orig_prompt + "This is an example for a response..." + golden_response); the student
# is the unconditioned base model wrapped with LoRA. Student samples its own completions
# on-policy from the bare prompt; per-token forward KL between teacher and student over
# those completions trains only the LoRA adapter. Teacher weights are frozen (no grads,
# no EMA sync). Uses DistilTrainer/DistilConfig from this repo (loss not reimplemented).
# vLLM, DeepSpeed, FSDP, and bf16 are disabled for MPS compatibility.

import argparse
import json
import os
from pathlib import Path
from string import Template

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from distil_trainer import DistilTrainer
from distil_config import DistilConfig


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
HOLDOUT_PATH = Path(
    "/Users/kyrylogy/Projects/University/WS25/Thesis/Self-Distillation/data/tooluse_data/train_subset_holdout_indices.json"
)
TRAIN_DATA_PATH = "data/tooluse_data/train_data"

# Byte-identical to main.py lines 30-37 (preserves leading newline inside the triple-quoted string).
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SDFT LoRA trainer (MPS-friendly)")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, truncate the (post-holdout, post-shuffle) training set to this many rows.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Hard cap on optimizer steps; passed to DistilConfig.max_steps so the run terminates on a laptop.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for trainer checkpoints, logs, and the final LoRA adapter.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace id used for BOTH student (LoRA-wrapped) and teacher (frozen).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="LoRA learning rate. 1e-4 is the standard Qwen2.5 LoRA starting point.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank. 16 balances expressivity and memory on a 3B model.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling). Convention alpha = 2*r so the effective scale is 2.0.",
    )
    parser.add_argument(
        "--no_holdout_filter",
        action="store_true",
        help="Skip the 100-row holdout exclusion. Use ONLY for debugging.",
    )
    parser.add_argument(
        "--ref_model_mixup_alpha",
        type=float,
        default=0.01,
        help="EMA mixup alpha for the teacher (ref) model; matches main.py default.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Holdout filter
# ---------------------------------------------------------------------------
def filter_holdout(raw_dataset, no_holdout_filter: bool):
    """Drop the 100 holdout indices BEFORE format_example/shuffle so the
    index space matches the on-disk source_dataset of HOLDOUT_PATH."""
    if no_holdout_filter:
        print(f"[holdout] --no_holdout_filter set; keeping all {len(raw_dataset)} rows")
        return raw_dataset
    with open(HOLDOUT_PATH, "r") as f:
        meta = json.load(f)
    holdout = set(meta["indices"])
    assert meta["source_rows"] == len(raw_dataset), (
        f"holdout file expects {meta['source_rows']} rows but dataset has {len(raw_dataset)}; "
        f"refusing to filter against a mismatched index space"
    )
    keep_idx = [i for i in range(len(raw_dataset)) if i not in holdout]
    filtered = raw_dataset.select(keep_idx)
    print(f"[holdout] dropped {len(holdout)} indices; {len(raw_dataset)} -> {len(filtered)} rows")
    return filtered


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_tooluse_dataset_filtered(seed: int, no_holdout_filter: bool, max_samples):
    """Load tooluse train_data, drop holdout indices, format prompts, shuffle,
    optionally truncate to max_samples."""
    raw_dataset = load_from_disk(TRAIN_DATA_PATH)
    print(f"[dataset] loaded {len(raw_dataset)} raw rows from {TRAIN_DATA_PATH}")

    raw_dataset = filter_holdout(raw_dataset, no_holdout_filter)

    def format_example(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [
                {
                    "role": "user",
                    "content": TEACHER_TEMPLATE.substitute(
                        orig_content=example["prompt"],
                        output_text="\n".join(example["golden_response"]),
                    ),
                }
            ],
        }

    formatted = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    formatted = formatted.shuffle(seed=seed)

    if max_samples is not None:
        n = min(max_samples, len(formatted))
        formatted = formatted.select(range(n))
        print(f"[dataset] truncated to --max_samples={max_samples}; final size {len(formatted)}")

    return formatted


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_models(model_name: str):
    """Load student and teacher (both fp32, same base id), freeze teacher, load tokenizer."""
    print(f"[models] loading student from {model_name} (fp32)")
    student = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    print(f"[models] loading teacher from {model_name} (fp32)")
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Defensive freeze: requires_grad=False on every teacher param + eval mode.
    # DistilTrainer.accelerator.prepare_model(ref_model, evaluation_mode=True) does NOT
    # set requires_grad=False, so we do it here to keep autograd graph minimal.
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return student, teacher, tokenizer


# ---------------------------------------------------------------------------
# LoRA wrap
# ---------------------------------------------------------------------------
def apply_lora(student, lora_r: int, lora_alpha: int):
    """Wrap student with LoRA. Targets Qwen2 attention + MLP projections."""
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    student_peft = get_peft_model(student, lora_cfg)
    student_peft.print_trainable_parameters()
    return student_peft


# ---------------------------------------------------------------------------
# Param-count logging
# ---------------------------------------------------------------------------
def log_param_counts(student_peft, teacher, train_dataset_size: int):
    trainable = sum(p.numel() for p in student_peft.parameters() if p.requires_grad)
    total_student = sum(p.numel() for p in student_peft.parameters())
    total_teacher = sum(p.numel() for p in teacher.parameters())
    teacher_grad = sum(p.numel() for p in teacher.parameters() if p.requires_grad)

    print(f"[params] student trainable params: {trainable:,}")
    print(f"[params] student total params:     {total_student:,}")
    print(f"[params] teacher total params:     {total_teacher:,}")
    print(f"[params] teacher trainable params: {teacher_grad:,} (must be 0)")
    print(f"[params] train_dataset_size_after_filter: {train_dataset_size}")

    assert trainable > 0, "LoRA wrap produced 0 trainable parameters — target_modules likely mismatched"
    assert teacher_grad == 0, "Teacher has trainable parameters — freeze failed"


# ---------------------------------------------------------------------------
# DistilConfig assembly
# ---------------------------------------------------------------------------
def build_distil_config(args) -> DistilConfig:
    """Build DistilConfig with all MPS-safe overrides per the design spec.

    Note on generate_from_teacher: with use_vllm=False the HF generate path at
    distil_trainer.py:1252-1258 ALWAYS samples from the student, regardless of
    this flag. We pin it False so SDFT on-policy semantics are explicit.
    """
    return DistilConfig(
        # vLLM completely disabled (CUDA-only).
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        # Precision: fp32 on MPS for stable KL/log_softmax math.
        bf16=False,
        fp16=False,
        # Batch / step budget — laptop scale.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=512,
        max_completion_length=256,
        num_generations=1,
        num_iterations=1,
        max_steps=args.max_steps,
        num_train_epochs=1,
        # Optimizer schedule.
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1,
        # Logging / checkpointing.
        logging_steps=1,
        save_steps=1_000_000,  # effectively disabled; we save the adapter manually.
        report_to="none",
        log_completions=False,
        # Teacher: NO EMA sync under LoRA.
        # main.py uses sync_ref_model=True with alpha=0.01 (TR-DPO-style EMA pull of
        # teacher toward student) — but that callback iterates model.parameters() and
        # tries shape-aligned mul_/add_ against ref_model.parameters(). Under PEFT the
        # student's parameter list includes LoRA A/B matrices that have no counterpart
        # on the teacher, causing a tensor-shape mismatch at on_step_end. Also: with
        # LoRA the student's base weights are frozen, so EMA-mixing teacher-base
        # toward student-base would be a no-op even if it didn't crash. Disabling.
        sync_ref_model=False,
        num_loss_tokens_to_skip=3,
        # KL losses.
        alpha=0,  # forward KL(teacher || student); SDFT default.
        beta=0,   # disable kl-to-base regularizer; keeps ref_model strictly teacher-routed.
        # MPS-safe attention / cache.
        use_transformers_paged=False,
        cache_implementation=None,
        # Memory.
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        # Dataset plumbing — must keep 'prompt' / 'teacher_prompt' columns alive.
        remove_unused_columns=False,
        # Determinism / output.
        seed=args.seed if hasattr(args, "seed") else 42,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    # Inject a fixed seed (not a CLI flag per spec, but DistilConfig needs one).
    args.seed = 42

    # 1) Models.
    student, teacher, tokenizer = build_models(args.model_name)

    # 2) LoRA wrap student only.
    student_peft = apply_lora(student, args.lora_r, args.lora_alpha)

    # 3) Dataset (load -> holdout filter -> format -> shuffle -> optional truncate).
    train_dataset = load_tooluse_dataset_filtered(
        seed=args.seed,
        no_holdout_filter=args.no_holdout_filter,
        max_samples=args.max_samples,
    )

    # 4) Param-count sanity check.
    log_param_counts(student_peft, teacher, len(train_dataset))

    # 5) DistilConfig.
    config = build_distil_config(args)

    # 6) Trainer. We pass an EXPLICIT teacher object so distil_trainer.py:405
    #    self.ref_model = ref_model takes precedence over the PEFT-None auto-path
    #    at distil_trainer.py:410-412.
    trainer = DistilTrainer(
        model=student_peft,
        ref_model=teacher,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Defensive assertions against future TRL refactors.
    assert trainer.ref_model is not None, "ref_model was nulled by trainer — teacher would be student-base"
    assert trainer.ref_model is not trainer.model.get_base_model(), (
        "ref_model is the same object as the student's base model — teacher routing collapsed"
    )

    # 7) Train.
    trainer.train()

    # 8) Save LoRA adapter (defensive: in addition to trainer's own save_model).
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    student_peft.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter written to {adapter_dir}")


if __name__ == "__main__":
    main()
