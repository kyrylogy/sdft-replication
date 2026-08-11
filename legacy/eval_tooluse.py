import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import re
from collections import Counter
from string import Template
from tqdm import tqdm


# Models we evaluate on ToolAlpaca. Short keys can be passed via --model_path
# and are resolved to HF hub ids below; full HF ids or local paths also work.
MODEL_REGISTRY = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",   # base/ceiling cells run with this
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",   # paper anchor (42% reference)
    "qwen3-4b":   "Qwen/Qwen3-4B",               # newer-gen comparison; adjust suffix when known
    # Calibration checkpoints — reference SDFT-trained 7B models for validating
    # our eval pipeline against known numbers (expect ~70 / ~70 / ~42.2).
    "sdft-7b-improbable": "improbableaimit/sdft-tooluse-7b",
    "sdft-7b-kickit":     "KickItLikeShika/qwen-2.5-7b-instruct-sdft-tooluse",
}


def resolve_model_id(model_path: str) -> str:
    """Map MODEL_REGISTRY short name -> HF hub id; pass through otherwise."""
    return MODEL_REGISTRY.get(model_path, model_path)


# Mirrors TEACHER_TEMPLATE in experiment.py — kept here so eval_tooluse.py
# stays self-contained. If experiment.py changes its template, change this too.
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


def _format_demo_from_row(row):
    """Build the matched-per-row demonstration string for teacher-ceiling.

    Prefer the dataset's own golden_response (real Thought+Action+Action Input
    strings, what the SDFT trainer in main.py feeds the teacher). Fall back to
    synthesizing Action+Input from golden_answer when golden_response is
    absent (e.g. data/tooluse_data/eval_data has no golden_response column).
    Returns (demo_text, source_tag).
    """
    gr = row.get('golden_response')
    if gr:
        return "\n".join(gr), "golden_response"
    parts = []
    for step in row['golden_answer']:
        parts.append(f"Action: {step['Action']}\nAction Input: {step['Action_Input']}")
    return "\n".join(parts), "synthesized_from_golden_answer"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on tooluse test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF hub id, local path, or MODEL_REGISTRY short key "
                             "(e.g. 'qwen2.5-3b', 'qwen2.5-7b', 'qwen3-4b').")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens to generate (default 2048 — the reproduction "
                             "report's reference protocol; earlier baselines used 1024).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--engine", type=str, default="auto",
                        choices=["auto", "vllm", "hf"],
                        help="Generation backend. 'auto' = vllm if CUDA else hf.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for hf engine: cuda | mps | cpu. Auto-detected if unset.")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"],
                        help="Dtype for hf engine. Defaults: cuda/mps=bfloat16, cpu=float32.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on number of eval samples (useful for smoke tests).")
    parser.add_argument("--eval_data", type=str,
                        default="data/tooluse_data/eval_data",
                        help="Path to an arrow dataset with at least {prompt, golden_answer}; "
                             "if it also has golden_response, --teacher_ceiling uses that "
                             "(matching the SDFT trainer's teacher condition).")
    parser.add_argument("--teacher_ceiling", action="store_true",
                        help="Wrap each prompt with TEACHER_TEMPLATE using that row's own "
                             "matched demonstration. Prefers golden_response when the dataset "
                             "has it; falls back to synthesis from golden_answer otherwise.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional PEFT/LoRA adapter directory to wrap on top of "
                             "--model_path (the base). HF engine only; vLLM path rejects it. "
                             "When set, tokenizer is loaded from the adapter dir if present.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                        help="vLLM engine memory budget (fraction of GPU). Default 0.5 fits a "
                             "shared 40 GB card with a ~15 GB orphan. Drop to 0.4/0.45 if vLLM "
                             "init reports 'less than desired'; raise to 0.7/0.8 on a clean card.")
    parser.add_argument("--no_enforce_eager", action="store_true",
                        help="Disable vLLM's enforce_eager. Default is ON because CUDA-graph "
                             "capture trips the Python.h include path on Py 3.12 in this image.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log eval accuracy + config to wandb as a summary run.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project (falls back to $WANDB_PROJECT).")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name (defaults to output_dir basename).")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional wandb group tag (e.g. arm name) so all "
                             "3 evals for one arm group in the UI.")
    return parser.parse_args()


def resolve_engine(engine: str) -> str:
    if engine == "auto":
        return "vllm" if torch.cuda.is_available() else "hf"
    return engine


def resolve_device(device):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name, device):
    if dtype_name is not None:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def load_vllm_model_and_tokenizer(model_path, gpu_memory_utilization=0.5, enforce_eager=True):
    """Load model using vLLM and tokenizer from the given path.

    Defaults are set for a shared 40 GB A100 with ~24 GB free (a ~15 GB
    orphan eats the rest): 0.5 utilization = ~20 GB, fits inside the
    free budget while leaving room for the KV cache on short prompts.
    enforce_eager=True skips CUDA-graph capture which is the only
    reasonable choice on Py 3.12 here — graph capture trips the
    Python.h include path that this image doesn't have.
    """
    from vllm import LLM  # lazy import: vllm is CUDA-only
    print(f"Loading model from {model_path} via vLLM "
          f"(gpu_memory_utilization={gpu_memory_utilization}, enforce_eager={enforce_eager})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_hf_model_and_tokenizer(model_path, device, dtype, adapter_path=None):
    """Load model using HF transformers (used on MPS/CPU and as a vLLM-free fallback on CUDA).

    If adapter_path is given, wrap the base from model_path with the LoRA/PEFT
    adapter saved there. Tokenizer is loaded from the adapter dir when it carries
    one (PEFT save_pretrained writes the tokenizer alongside the adapter); else
    from model_path. This is the path for evaluating a trained adapter.
    """
    from transformers import AutoModelForCausalLM
    print(f"Loading base from {model_path} via HF transformers on {device} ({dtype})")
    tokenizer_src = adapter_path if (adapter_path and os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json"))) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if adapter_path:
        from peft import PeftModel
        print(f"Wrapping with PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_test_data(tokenizer, eval_data_path, teacher_ceiling=False):
    """Load and prepare tooluse test dataset.

    If teacher_ceiling=True, wrap each row's prompt with TEACHER_TEMPLATE using
    that row's own matched demonstration (golden_response when available,
    else synthesized from golden_answer) — the SDFT teacher condition.
    Returns (data, demo_source_tag) where demo_source_tag identifies which
    demo format was used (None when teacher_ceiling=False).
    """
    data = load_from_disk(eval_data_path).to_list()

    demo_source = None
    for example in data:
        if teacher_ceiling:
            demo_text, src = _format_demo_from_row(example)
            if demo_source is None:
                demo_source = src
            content = TEACHER_TEMPLATE.substitute(
                orig_content=example['prompt'],
                output_text=demo_text,
            )
        else:
            content = example['prompt']
        example['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}],
            tokenize=False,
            add_generation_prompt=True
        )

    return data, demo_source


def generate_responses_vllm(llm, tokenizer, prompts, max_new_tokens=1024, temperature=0.0):
    """Generate responses from the model using vLLM."""
    from vllm import SamplingParams  # lazy import
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def generate_responses_hf(model, tokenizer, prompts, device, max_new_tokens=1024, temperature=0.0):
    """Generate responses with HF transformers (batch_size=1 for MPS memory safety)."""
    print(f"Generating responses for {len(prompts)} prompts (hf, device={device})...")
    responses = []
    do_sample = temperature > 0
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def extract_actions(text):
    """Extract all actions from model response."""
    return re.findall(r'Action:\s*(\w+)', text)


def extract_action_inputs(text):
    """Extract and merge all action inputs from model response."""
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            continue
    return combined_dict


def evaluate_correctness(responses, golden_answers):
    """
    Evaluate if responses match the golden answers.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    
    for response, golden_answer in zip(responses, golden_answers):
        # Extract predicted actions and inputs
        pred_actions = extract_actions(response)
        pred_inputs = extract_action_inputs(response)
        
        # Extract ground truth actions and inputs
        gt_actions = [item['Action'] for item in golden_answer]
        gt_inputs = {}
        for item in golden_answer:
            try:
                gt_inputs.update(json.loads(item['Action_Input']))
            except:
                pass
        
        # Check if both actions and inputs match
        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = pred_inputs == gt_inputs
        
        results.append(1 if (actions_match and inputs_match) else 0)
    
    return results


def main():
    args = parse_args()

    # Resolve MODEL_REGISTRY short key; remember original for traceability.
    model_short_name = args.model_path if args.model_path in MODEL_REGISTRY else None
    args.model_path = resolve_model_id(args.model_path)

    engine = resolve_engine(args.engine)
    device = resolve_device(args.device) if engine == "hf" else "cuda"
    dtype = resolve_dtype(args.dtype, device) if engine == "hf" else torch.bfloat16
    print(f"Engine: {engine} | device: {device} | dtype: {dtype}")

    # Load model + tokenizer
    if engine == "vllm":
        if args.adapter_path:
            raise SystemExit("--adapter_path is HF-engine only; pass --engine hf or merge the adapter first.")
        llm, tokenizer = load_vllm_model_and_tokenizer(
            args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=not args.no_enforce_eager,
        )
    else:
        model, tokenizer = load_hf_model_and_tokenizer(args.model_path, device, dtype, adapter_path=args.adapter_path)

    # Load and (optionally) cap test data
    test_data, demo_source = load_test_data(
        tokenizer,
        eval_data_path=args.eval_data,
        teacher_ceiling=args.teacher_ceiling,
    )
    print(f"Eval data: {args.eval_data} ({len(test_data)} rows)")
    if args.teacher_ceiling:
        print(f"Teacher-ceiling mode: demo source = {demo_source}")
    if args.max_samples is not None:
        test_data = test_data[: args.max_samples]
        print(f"Capped eval set to first {len(test_data)} samples (--max_samples).")

    prompts = [example['prompt'] for example in test_data]
    golden_answers = [example['golden_answer'] for example in test_data]

    # Generate responses
    if engine == "vllm":
        responses = generate_responses_vllm(
            llm, tokenizer, prompts, args.max_new_tokens, args.temperature
        )
    else:
        responses = generate_responses_hf(
            model, tokenizer, prompts, device, args.max_new_tokens, args.temperature
        )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, golden_answers)
    accuracy = float(np.mean(scores)) if scores else 0.0

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": accuracy,
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "model_short_name": model_short_name,
            "vllm_gpu_memory_utilization": args.gpu_memory_utilization if engine == "vllm" else None,
            "vllm_enforce_eager": (not args.no_enforce_eager) if engine == "vllm" else None,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "engine": engine,
            "device": device,
            "dtype": str(dtype),
            "max_samples": args.max_samples,
            "teacher_ceiling": args.teacher_ceiling,
            "eval_data": args.eval_data,
            "demo_source": demo_source,
            "adapter_path": args.adapter_path,
        }
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save responses for inspection / failure-mode analysis
    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": test_data[i]['prompt'],
                "response": responses[i],
                "golden_answer": golden_answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")

    if args.wandb:
        try:
            import wandb
            project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "sdft-replication"
            run_name = args.wandb_run_name or os.path.basename(output_dir.rstrip("/"))
            wandb.init(
                project=project,
                name=run_name,
                group=args.wandb_group,
                job_type="eval",
                config=results_to_save["config"],
                reinit=True,
            )
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/num_correct": int(sum(scores)),
                "eval/num_total": len(scores),
            })
            wandb.summary["accuracy"] = accuracy
            wandb.summary["num_correct"] = int(sum(scores))
            wandb.summary["num_total"] = len(scores)
            wandb.summary["eval_results_json"] = output_path
            wandb.finish()
            print(f"[wandb] logged eval run '{run_name}' to project '{project}'")
        except Exception as e:
            print(f"[wandb] logging skipped due to error: {e}")


if __name__ == "__main__":
    main()
