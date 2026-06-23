"""Tiny SDFT smoke-test on Qwen2.5-3B-Instruct.

Purpose: a minimal driver to load the model, take a few tooluse rows as a
placeholder dataset, and run a couple of optimizer steps with DistilTrainer
so you can confirm the pipeline works end-to-end and iterate on it.

Run from this directory:

    python experiment.py

Toggles you'll likely change:
    --max-samples       how many rows to use (default 8)
    --max-steps         hard cap on optimizer steps (default 2)
    --use-vllm          enable vLLM generation (CUDA only)
    --dataset           tooluse | science
"""

import argparse
import torch
from datasets import load_from_disk
from string import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_trainer import DistilTrainer
from distil_config import DistilConfig


TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        # MPS bf16 is supported on recent PyTorch + Apple Silicon, but fp32 is the
        # safest default for a smoke test where speed isn't the point.
        return "mps", torch.float32
    return "cpu", torch.float32


def load_tiny_tooluse(n: int, seed: int):
    ds = load_from_disk("data/tooluse_data/train_data")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def fmt(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "teacher_prompt": [{
                "role": "user",
                "content": TEACHER_TEMPLATE.substitute(
                    orig_content=ex["prompt"],
                    output_text="\n".join(ex["golden_response"]),
                ),
            }],
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def load_tiny_science(n: int, seed: int):
    ds = load_from_disk("data/science_data/train_data")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def fmt(ex):
        return {
            "prompt": ex["messages"],
            "teacher_prompt": [
                ex["messages"][0],
                {
                    "role": "user",
                    "content": TEACHER_TEMPLATE.substitute(
                        orig_content=ex["messages"][1]["content"],
                        output_text=ex["output_text"],
                    ),
                },
            ],
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="tooluse", choices=["tooluse", "science"])
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--output-dir", default="./out-sdft-smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-vllm", action="store_true",
                   help="Enable vLLM for generation (CUDA only).")
    args = p.parse_args()

    device, dtype = pick_device_and_dtype()
    print(f"[experiment] device={device} dtype={dtype} model={args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    loader = load_tiny_tooluse if args.dataset == "tooluse" else load_tiny_science
    train_ds = loader(args.max_samples, args.seed)
    print(f"[experiment] dataset={args.dataset} rows={len(train_ds)}")

    cfg = DistilConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        # Tiny everything — this is a smoke test, not a training run.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        max_prompt_length=512,
        max_completion_length=256,
        num_iterations=1,
        num_generations=1,
        logging_steps=1,
        save_steps=10_000,  # don't save during smoke test
        max_grad_norm=1.0,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        # Generation backend
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else "server",
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=args.use_vllm,
        vllm_tensor_parallel_size=1,
        vllm_importance_sampling_correction=args.use_vllm,
        # SDFT specifics
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=0.01,
        num_loss_tokens_to_skip=3,
        log_completions=False,
        report_to="none",
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    print("[experiment] done")


if __name__ == "__main__":
    main()
`´´