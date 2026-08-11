# SDFT (Self-Distillation Fine-Tuning) LoRA trainer — MPS and CUDA.
# The teacher is the SAME base model conditioned in-context on a golden demonstration
# (orig_prompt + "This is an example for a response..." + golden_response); the student
# is the unconditioned base model wrapped with LoRA. Student samples its own completions
# on-policy from the bare prompt; per-token forward KL between teacher and student over
# those completions trains only the LoRA adapter. Uses DistilTrainer/DistilConfig from
# this repo (loss not reimplemented).
#
# Two teacher modes:
#   default                : teacher base frozen, no EMA. This is the A.3
#                            underperforming arm; use it as the lower-bound SDFT.
#   --teacher_adapter_ema  : teacher gets its OWN LoRA (zero-init), and the
#                            student's LoRA is EMA-mixed into the teacher's LoRA
#                            every step. Recovers the paper's A.3 EMA-of-student
#                            teacher mechanism under LoRA. This is the
#                            paper-faithful arm.
#
# Set --generate_from_teacher (with --use_vllm) for the ONLINE SFT ablation.
# For CLASSIC offline SFT (cross-entropy on golden_response, no teacher), use
# train_sft_lora.py instead.
#
# Invocations:
#   # MPS (laptop) — single process
#   python train_sdft_lora.py --max_samples 8 --max_steps 1 --output_dir /tmp/sdft_smoke
#
#   # 1 GPU (CUDA) — single process, identical args to MPS
#   python train_sdft_lora.py --max_steps 200 --output_dir runs/sdft_1gpu
#
#   # 2–4 GPU DDP — accelerate handles data-parallel sharding of the batch
#   accelerate launch --multi_gpu --num_processes 4 \
#     train_sdft_lora.py --max_steps 200 --output_dir runs/sdft_4gpu
#
# Multi-GPU notes: no code changes are required for DDP because DistilTrainer is a
# TRL/HF Trainer subclass — accelerate replicates the model on each rank and shards
# the batch automatically. Effective batch size = per_device_train_batch_size *
# num_processes * gradient_accumulation_steps. LoRA's trainable params are small
# (~0.24% of the 3B base), but memory is dominated by holding student + teacher
# resident: ~12 GB for two 3B copies in bf16, ~28 GB for two 7B copies. So 3B fits
# on one 24 GB GPU comfortably; 7B wants 40 GB+ on one GPU or two 24 GB GPUs.

import argparse
import json
import os
from pathlib import Path
from string import Template

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

from distil_trainer import DistilTrainer
from distil_config import DistilConfig


# ---------------------------------------------------------------------------
# LoRA-aware EMA callback — recovers the paper's Appendix-A.3 EMA-of-student
# teacher mechanism under LoRA. Frozen-base alone is the A.3 underperforming
# arm; this callback updates the teacher's LoRA A/B from the student's LoRA
# A/B so the teacher actually tracks the student rather than staying fixed.
# Replaces MemoryEfficientSyncRefModelCallback (which iterates full
# named_parameters() and crashes on the LoRA A/B vs base shape mismatch).
# ---------------------------------------------------------------------------
class LoRAEMACallback(TrainerCallback):
    def __init__(self, student_module, teacher_module, alpha: float):
        self.student_module = student_module
        self.teacher_module = teacher_module
        self.alpha = float(alpha)
        student_lora = {n: p for n, p in student_module.named_parameters() if "lora_" in n}
        teacher_lora = {n: p for n, p in teacher_module.named_parameters() if "lora_" in n}
        self.pairs = []
        for name, sp in student_lora.items():
            tp = teacher_lora.get(name)
            if tp is None or tp.shape != sp.shape:
                continue
            self.pairs.append((sp, tp))
        print(
            f"[ema] LoRAEMACallback matched {len(self.pairs)} LoRA parameter pairs "
            f"(student_lora={len(student_lora)}, teacher_lora={len(teacher_lora)}); alpha={self.alpha}"
        )
        assert self.pairs, "LoRAEMACallback found no matching LoRA pairs — teacher LoRA wrap failed?"

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        for sp, tp in self.pairs:
            tp.data.mul_(1.0 - self.alpha).add_(sp.data, alpha=self.alpha)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent
HOLDOUT_PATH = _REPO / "data/tooluse_data/train_subset_holdout_indices.json"
TRAIN_DATA_PATH = _REPO / "data/tooluse_data/train_data"

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
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffle and DistilConfig.",
    )
    # ---- CUDA / cluster knobs (default to MPS-safe values) ----
    parser.add_argument("--use_vllm", action="store_true",
                        help="Enable vLLM for student rollouts. Requires CUDA. Faster than HF generate.")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "server"],
                        help="vLLM placement. 'colocate' shares the training GPU (paper-default in main.py); "
                             "'server' expects a separate vLLM server. Inert when --use_vllm is off.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="Fraction of GPU memory vLLM may use. Paper main.py uses 0.3 to leave room "
                             "for the training process + teacher. Lower further on shared GPUs.")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true",
                        help="Let vLLM sleep during gradient updates to free GPU memory (paper-default in main.py). "
                             "Recommended when --vllm_mode=colocate.")
    parser.add_argument("--vllm_importance_sampling_correction", action="store_true",
                        help="Enable IS correction when vLLM and training step rollouts can diverge. Pair with --use_vllm.")
    parser.add_argument("--bf16", action="store_true",
                        help="Load models in bf16 and run training in bf16. CUDA-only — keep off on MPS.")
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Token cap on student completions. 256 was the MPS budget; bump to 1024 on CUDA.")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Token cap on prompts. 512 was the MPS budget; bump to 1024 on CUDA.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=None,
                        help="If set, overrides --max_steps. Use for cluster runs (e.g. 2.0 per paper).")
    parser.add_argument("--enable_input_require_grads", action="store_true",
                        help="Defensive call after LoRA wrap when gradient_checkpointing=True + PEFT + DistilTrainer. "
                             "Flip on if step-1 LoRA grad_norm comes back zero.")
    parser.add_argument("--generate_from_teacher", action="store_true",
                        help="Use teacher (not student) for rollouts -> trainer becomes ONLINE SFT. "
                             "Requires --use_vllm (HF generate path always uses student, regardless of this flag).")
    parser.add_argument("--teacher_adapter_ema", action="store_true",
                        help="Wrap the teacher with its OWN LoRA (zero-initialized) and EMA-mix the "
                             "student's LoRA into it after every step. Recovers the paper's A.3 "
                             "EMA-of-student teacher under LoRA — otherwise the teacher stays frozen "
                             "(the A.3 underperforming ablation arm).")
    parser.add_argument("--ema_alpha", type=float, default=0.01,
                        help="EMA mixup rate for --teacher_adapter_ema (paper-aligned default 0.01).")
    # ---- Metric tracking ----
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard"],
                        help="Where to ship training metrics. 'wandb' uses WANDB_API_KEY env var "
                             "or ~/.netrc from `wandb login`. 'tensorboard' writes event files "
                             "to <output_dir>/runs/.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for wandb/tensorboard. Defaults to output_dir basename.")
    parser.add_argument("--wandb_project", type=str, default="sdft-replication",
                        help="W&B project name. Sets WANDB_PROJECT before Trainer init.")
    # ---- Checkpointing / resume (so a 5-8h 7B kill doesn't lose the whole arm) ----
    parser.add_argument("--save_steps", type=int, default=1_000_000,
                        help="HF Trainer checkpoint interval. Default effectively disabled (matches "
                             "MPS-era behavior). On cluster pass e.g. 50 so a mid-arm kill leaves a "
                             "resumable checkpoint at <output_dir>/checkpoint-<N>.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Keep at most N most-recent checkpoints (older ones are GC'd by Trainer).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir or 'auto' to pick the latest under --output_dir. "
                             "When set, trainer.train() resumes from there instead of starting fresh.")
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
def build_models(model_name: str, bf16: bool = False):
    """Load student and teacher (same base id), freeze teacher, load tokenizer."""
    dtype = torch.bfloat16 if bf16 else torch.float32
    print(f"[models] loading student from {model_name} ({dtype})")
    student = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    print(f"[models] loading teacher from {model_name} ({dtype})")
    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    # DistilTrainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
    # puts the model in eval mode but does NOT freeze grads — do it here so the
    # teacher cannot accidentally accumulate gradients via the KL backward graph.
    for p in teacher.parameters():
        p.requires_grad_(False)

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
    # If --num_train_epochs is set, hand max_steps=-1 to disable the step cap.
    max_steps_val = -1 if args.num_train_epochs is not None else args.max_steps
    num_epochs_val = args.num_train_epochs if args.num_train_epochs is not None else 1.0
    return DistilConfig(
        # vLLM rollouts. CUDA-only; flipped by --use_vllm on cluster.
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
        generate_from_teacher=args.generate_from_teacher,  # True => online SFT (requires vllm)
        # Precision: fp32 on MPS for stable KL/log_softmax math; bf16 on CUDA.
        bf16=args.bf16,
        fp16=False,
        # Batch / step budget — caller decides.
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=1,
        num_iterations=1,
        max_steps=max_steps_val,
        num_train_epochs=num_epochs_val,
        # Optimizer schedule.
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1,
        # Logging / checkpointing.
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps" if args.save_steps < 1_000_000 else "no",
        report_to=args.report_to,
        run_name=args.run_name or os.path.basename(args.output_dir.rstrip("/")),
        log_completions=False,
        # Teacher: EMA sync OFF. NOTE — this is a DEVIATION from the paper.
        # Appendix A.3 ablates teacher choices and recommends EMA-of-student
        # (which is what main.py wires via sync_ref_model=True, alpha=0.01).
        # We disable it here only because the callback iterates
        # model.parameters() against ref_model.parameters() and crashes on the
        # LoRA A/B shape mismatch. Even with that patched, EMA-mixing
        # teacher_base ← student_base is a no-op under LoRA (student's base is
        # frozen). A proper paper-faithful LoRA SDFT would need a different
        # mechanism (teacher LoRA + adapter EMA, or merged-weight EMA).
        # See WORK_LOG.md for details.
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
        seed=args.seed,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # SFT-LoRA mode requires vLLM — HF generate path always uses student
    # (distil_trainer.py:1252-1258), regardless of generate_from_teacher.
    if args.generate_from_teacher and not args.use_vllm:
        raise SystemExit(
            "--generate_from_teacher requires --use_vllm; the HF generate path "
            "always samples from the student. Pass both flags for SFT-LoRA mode."
        )

    # Wire wandb project / mode BEFORE the Trainer touches the WandbCallback.
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        # Save full configs as wandb artifacts; quiet the launch spam.
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print(f"[wandb] project={args.wandb_project} run_name={args.run_name or 'auto'}")

    # 1) Models.
    student, teacher, tokenizer = build_models(args.model_name, bf16=args.bf16)

    # 2) LoRA wrap student.
    student_peft = apply_lora(student, args.lora_r, args.lora_alpha)

    # 2b) If --teacher_adapter_ema, also LoRA-wrap the teacher (default init
    #     puts B=0, so initial delta is zero → teacher initially behaves as
    #     the bare base + ICL demo, identical to the frozen-teacher mode).
    #     The student's LoRA will then be EMA-mixed into the teacher's LoRA
    #     each step via LoRAEMACallback. Teacher base stays frozen.
    teacher_peft = None
    if args.teacher_adapter_ema:
        teacher_peft = apply_lora(teacher, args.lora_r, args.lora_alpha)
        # Teacher LoRA params should NOT receive gradients — EMA only touches .data.
        for p in teacher_peft.parameters():
            p.requires_grad_(False)
        print("[ema] teacher LoRA-wrapped (B=0 init, no grads); EMA will track student LoRA")

    # Defensive: when gradient_checkpointing=True + PEFT + a custom Trainer
    # subclass (DistilTrainer), HF's auto-detection of PEFT may not fire and
    # input requires_grad gets dropped. Flip --enable_input_require_grads if
    # step-1 LoRA grad_norm comes back zero on cluster.
    if args.enable_input_require_grads:
        student_peft.enable_input_require_grads()
        print("[lora] called student_peft.enable_input_require_grads()")

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
    #    at distil_trainer.py:410-412. With --teacher_adapter_ema, pass the
    #    PEFT-wrapped teacher so the EMA callback can locate the LoRA params.
    ref_model_for_trainer = teacher_peft if args.teacher_adapter_ema else teacher
    trainer = DistilTrainer(
        model=student_peft,
        ref_model=ref_model_for_trainer,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    if args.teacher_adapter_ema:
        trainer.add_callback(
            LoRAEMACallback(student_peft, teacher_peft, alpha=args.ema_alpha)
        )

    # Defensive assertions against future TRL refactors.
    assert trainer.ref_model is not None, "ref_model was nulled by trainer — teacher would be student-base"
    assert trainer.ref_model is not trainer.model.get_base_model(), (
        "ref_model is the same object as the student's base model — teacher routing collapsed"
    )

    # 7) Train (with optional resume from a prior checkpoint).
    resume_arg = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            # Pick the largest checkpoint-* dir under output_dir, if any.
            ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
            if ckpts:
                resume_arg = str(ckpts[-1])
                print(f"[resume] auto-detected latest checkpoint: {resume_arg}")
            else:
                print(f"[resume] --resume_from_checkpoint=auto: no checkpoint-* found in {args.output_dir}; fresh start.")
        else:
            resume_arg = args.resume_from_checkpoint
            print(f"[resume] using checkpoint: {resume_arg}")
    trainer.train(resume_from_checkpoint=resume_arg)

    # 8) Save LoRA adapter (rank-0 only under DDP, defensive on single-process too).
    if trainer.accelerator.is_main_process:
        adapter_dir = os.path.join(args.output_dir, "lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        student_peft.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[save] LoRA adapter written to {adapter_dir}")


if __name__ == "__main__":
    main()
