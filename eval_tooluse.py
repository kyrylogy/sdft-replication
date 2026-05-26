import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import re
from collections import Counter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on tooluse test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF hub id or local path of the model to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate")
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


def load_vllm_model_and_tokenizer(model_path, gpu_memory_utilization=0.8):
    """Load model using vLLM and tokenizer from the given path."""
    from vllm import LLM  # lazy import: vllm is CUDA-only
    print(f"Loading model from {model_path} via vLLM")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_hf_model_and_tokenizer(model_path, device, dtype):
    """Load model using HF transformers (used on MPS/CPU and as a vLLM-free fallback on CUDA)."""
    from transformers import AutoModelForCausalLM
    print(f"Loading model from {model_path} via HF transformers on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_test_data(tokenizer):
    """Load and prepare tooluse test dataset."""
    data_dir = 'data/tooluse_data/eval_data'
    data = load_from_disk(data_dir).to_list()
    
    # Format prompts
    for example in data:
        example['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': example['prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
    
    return data


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

    engine = resolve_engine(args.engine)
    device = resolve_device(args.device) if engine == "hf" else "cuda"
    dtype = resolve_dtype(args.dtype, device) if engine == "hf" else torch.bfloat16
    print(f"Engine: {engine} | device: {device} | dtype: {dtype}")

    # Load model + tokenizer
    if engine == "vllm":
        llm, tokenizer = load_vllm_model_and_tokenizer(args.model_path)
    else:
        model, tokenizer = load_hf_model_and_tokenizer(args.model_path, device, dtype)

    # Load and (optionally) cap test data
    test_data = load_test_data(tokenizer)
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
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "engine": engine,
            "device": device,
            "dtype": str(dtype),
            "max_samples": args.max_samples,
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


if __name__ == "__main__":
    main()
