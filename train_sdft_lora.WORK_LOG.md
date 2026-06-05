# train_sdft_lora.py — Work Log

## TL;DR
- Built a laptop/MPS-friendly SDFT trainer that wraps the student in LoRA, uses the same base model as a frozen-but-EMA-synced teacher conditioned on a golden-response demonstration, and reuses the repo's `DistilTrainer`/`DistilConfig` for the forward-KL loss — vLLM, DeepSpeed, FSDP, and bf16 are all disabled for MPS compatibility.
- Validation on the project venv failed at import time (`ModuleNotFoundError: No module named 'peft'`); the script never reached the model load or training loop. Once `peft` is installed, the script should be re-run end-to-end on MPS to confirm a non-zero loss tick.
- Next steps: `pip install peft` (and re-verify other LoRA deps), then smoke at `--max_samples 8 --max_steps 1`; if that ticks, scale to `--max_steps 20` and confirm the LoRA adapter is written to `<output_dir>/lora_adapter/`.

## What the SDFT trainer actually does

**"Teacher and student are the same model" — confirmed, with a sync-twist.** `main.py` lines 84-91 instantiate two independent `AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)` objects from the same HF id (default `Qwen/Qwen2.5-7B-Instruct`, line 18). They share weights only at init; thereafter they are distinct module instances passed as `model=` and `ref_model=` to `DistilTrainer` (`main.py:131-137`). The teacher is not in the optimizer and has `evaluation_mode=True` set via `accelerator.prepare_model` (`distil_trainer.py:555`), so it receives no gradients. However, `main.py` sets `sync_ref_model=True`, `ref_model_sync_steps=1`, `ref_model_mixup_alpha=0.01`, which registers `MemoryEfficientSyncRefModelCallback` (`distil_trainer.py:91-150, 557-558`). On every step `on_step_end` runs `ref_param.data.mul_(1.0 - 0.01).add_(model_param.data, alpha=0.01)` — an EMA pull of the teacher toward the student (TR-DPO-style). So: same architecture, no backprop on teacher, but weights are continuously mixed each step.

**"Teacher has context given" — confirmed, routed at the prompt level.** The dataset map in `main.py:39-42` emits two columns: `prompt` is the bare user message and `teacher_prompt` substitutes both `$orig_content` (the original prompt) and `$output_text` (`'\n'.join(example['golden_response'])`) into the template at `main.py:30-37` (`"This is an example for a response to the question:\n$output_text\n\nNow answer with a response of your own, including the thinking process."`). Inside the trainer, `distil_trainer.py:1322-1323` reads both keys; `1336` selects `generation_prompts = teacher_prompts if self.generate_from_teacher else prompts`; `1608` builds the student input `torch.cat([prompt_ids, completion_ids], dim=1)` and `1610` builds `teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)`. The student forward (`1615-1617`) sees only the bare prompt; the teacher forward (`1630-1633`) sees the demonstration-bearing prompt. Under `beta=0` (which `main.py` uses implicitly) the only place `self.ref_model` is called is line 1631 with `teacher_input_ids`.

**"KL divergence minimized" — forward KL, per-token, masked, student-on-policy by default.** The branch at `distil_trainer.py:1656-1676` reads `if self.alpha == 0: kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)`. PyTorch's `F.kl_div(input, target, log_target=True)` computes `exp(target)*(target - input)`, so with `input=student_logp, target=teacher_logp` and a sum over the vocab axis (`per_token_loss = kl_loss.sum(-1)`, line 1676) this is `sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))` = KL(teacher ‖ student) = forward KL. `DistilConfig` defaults `alpha=0.0` (`distil_config.py:486`). The loss is per-token (shape (B, T) after the vocab sum), masked by `loss_completion_mask` which excludes prompt tokens via `logits_to_keep=completion_ids.size(1)` (`1612`), excludes right-padding, and optionally zeros the first `num_loss_tokens_to_skip` completion tokens (`1600-1606`). Final reduction at `1687` is per-sequence masked mean over completion tokens, then batch mean. Samples are student on-policy under the default `generate_from_teacher=False` (`distil_config.py:493`): `_generate_and_score_completions` produces completions from the student model and, under vLLM, applies `vllm_importance_sampling_correction` to fix sampler/policy mismatch (`1454-1458, 1678-1682`).

**"Trivial because it's part of the SDFT repo" — mostly, but with two surprises.** Reusing `DistilTrainer`/`DistilConfig` is the right call — the loss, masking, importance-sampling correction, and per-token KL are all already wired. What surprised us during verification: (1) the "frozen teacher" intuition is wrong under `main.py`'s defaults — with `sync_ref_model=True, alpha=0.01`, the teacher moves every step; calling it "frozen" only describes gradients, not weights. (2) Only the vLLM path actually honors `generate_from_teacher=True`; both HF generate paths (`distil_trainer.py:1226, 1258`) always unwrap `self.model_wrapped` (the student). So if `use_vllm=False` (our case), `generate_from_teacher=True` would silently still sample from the student — a latent footgun. We set `generate_from_teacher=False` explicitly.

## Resource changes for laptop runnability
- **vLLM off (`use_vllm=False`)** — CUDA-only; top MPS blocker (`distil_trainer.py:51-52, 81-82, 445-522`). Forces fallback to the HF `model.generate` path (`1252-1258`) which works on MPS.
- **DeepSpeed / FSDP off** — `accelerate launch` without `--use_deepspeed`/`--fsdp`; `is_deepspeed_enabled=False` skips the `import deepspeed` paths inside `MemoryEfficientSyncRefModelCallback` (`distil_trainer.py:121-138`) and `prepare_deepspeed` (`549-551`).
- **LoRA on student only** — `peft.get_peft_model` wraps the student with `r=16, alpha=32, target_modules=[q/k/v/o/gate/up/down_proj], dropout=0.05, bias='none', task_type=CAUSAL_LM`. Teacher remains a bare `AutoModelForCausalLM` with `requires_grad_(False)` + `.eval()`.
- **bf16=False, fp16=False (fp32)** — MPS bf16 is unstable for matmul/SDPA; fp32 also gives cleaner KL/log_softmax math. Both student and teacher loaded with `torch_dtype=torch.float32`.
- **Prompt/completion length cuts** — `max_prompt_length=512` (halved from 1024), `max_completion_length=256` (quartered from 1024) to bound memory with student+teacher both resident.
- **batch=1, grad_accum=1, max_steps=20 default** — laptop budget; hard cap on optimizer steps so the run terminates regardless of dataset size.
- **Holdout filter on train data** — `filter_holdout()` drops 100 reserved-for-eval indices from `train_subset_holdout_indices.json` BEFORE `map/shuffle/select` so the index space matches the on-disk source dataset (`source_rows == len(raw_dataset)` asserted).
- **`gradient_checkpointing=True`** — inherited `DistilConfig` default; needed to fit student+teacher into 24GB MPS.
- **`use_transformers_paged=False`, `cache_implementation=None`** — avoid FlashAttention2/paged_attention (`distil_trainer.py:1208-1234`) and CUDA-specific static caches.
- **`report_to='none'`, `save_steps=1000000`** — disable wandb for offline laptop, disable mid-run checkpoints; final adapter saved manually via `student_peft.save_pretrained(<output_dir>/lora_adapter)`.
- **`beta=0` (kept)** — disables the kl-to-base-model regularizer branch (`distil_trainer.py:1463-1474`), so `self.ref_model` is only called with `teacher_input_ids` (line 1631), keeping student/teacher prompt routing strictly separated.

## Reviewer findings and resolution

| Severity | Location | Finding | Action |
|----------|----------|---------|--------|
| Critical (fidelity) | DistilConfig sync_ref_model trio | New file had `sync_ref_model=False`; `main.py` uses EMA-synced teacher with `alpha=0.01` every step | **Applied**: flipped to `sync_ref_model=True, ref_model_sync_steps=1, ref_model_mixup_alpha=args.ref_model_mixup_alpha (default 0.01)`; added `--ref_model_mixup_alpha` CLI arg mirroring `main.py:16` |
| Minor | Bugs (relative path, doc strings, dropout, holdout source_dataset, max_grad_norm int, dead hasattr) | Stylistic / defensive nits | Skipped per scope; reviewer marked overall "ship" |
| Low (fidelity) | Teacher dtype fp32 vs main.py bf16 | MPS lacks stable bf16 | Skipped — deliberate resource constraint |
| Low (fidelity) | `vllm_importance_sampling_correction=False` | No-op under `use_vllm=False` | Skipped — paired knob |
| Low (fidelity) | Explicit teacher `requires_grad_(False) + .eval()` | Does not change loss; defensive | Skipped |
| Low (fidelity) | LoRA vs full-FT student | Intended scope of this file | Skipped |
| Wiring | LoRA wraps student only; targets match Qwen2.5; adapter save_pretrained | All 6 checks pass | No action |

## Validation result

### First attempt — blocked at import

`python train_sdft_lora.py ...` exited with code **1** at import time: `ModuleNotFoundError: No module named 'peft'`. The project `.venv` was missing `peft`.

### Second attempt (after `pip install peft trl`) — caught a real bug, predicted by the open-questions section

The EMA-sync callback from `main.py` (`sync_ref_model=True, alpha=0.01`) is incompatible with PEFT: `MemoryEfficientSyncRefModelCallback.on_step_end` iterates `model.parameters()` (which under PEFT includes LoRA A/B matrices) and tries to `mul_/add_` shape-aligned against `ref_model.parameters()` (base-only on teacher). Crashed with `RuntimeError: The size of tensor a (256) must match the size of tensor b (4) at non-singleton dimension 0` — i.e., a LoRA `[256, 4]` matrix vs a `[256, 256]` teacher weight at the same iteration index. Training step **completed before the callback** (loss was computed, backward ran), so the SDFT mechanics themselves were correct; only the post-step EMA crashed.

### Third attempt (after disabling `sync_ref_model` under LoRA) — clean end-to-end

```
.venv/bin/python train_sdft_lora.py --max_samples 4 --max_steps 1 \
  --output_dir /tmp/sdft_lora_smoke2 --lora_r 4 --lora_alpha 8
```

- exit_code: **0**
- runtime: ~2:50 total (student load 6s + teacher load 20s + 1 train step 2:28)
- trainable params: 7,483,392 of 3,093,422,080 (0.24%) — LoRA on student only
- teacher trainable: 0 (confirmed frozen)
- holdout filter: 4046 → 3946 rows (100 indices dropped per manifest)
- **`loss = 0.2671`** (finite, non-zero)
- **`grad_norm = 6.65`** (LoRA params received gradients)
- `kl_approx = 0.118`, `entropy = 0.124`, `completions/mean_length = 64`
- LoRA adapter saved: `/tmp/sdft_lora_smoke2/lora_adapter/` (30 MB safetensors + tokenizer)
- Adapter reload check: `PeftModel.from_pretrained(base, adapter_dir)` succeeds, returns `PeftModelForCausalLM` class

### Fix applied to make it run

In `build_distil_config`:

```python
# Was:
# sync_ref_model=True, ref_model_sync_steps=1, ref_model_mixup_alpha=args.ref_model_mixup_alpha,
# Now:
sync_ref_model=False,  # disable under LoRA — see comment in file for why
```

Comment in the file explains: under LoRA the student's base weights are frozen, so the EMA mix toward the teacher's base is a no-op even if the shape mismatch were patched. This is a **fidelity gap vs `main.py`** that we accept as a cost of LoRA — see "Open questions" below.

## How to use the file

Laptop smoke (single step, 8 rows):
```bash
python train_sdft_lora.py \
  --max_samples 8 \
  --max_steps 1 \
  --output_dir /tmp/sdft_lora_smoke
```

Slightly bigger laptop run (20 steps, full holdout-filtered set, batch=1):
```bash
python train_sdft_lora.py \
  --max_steps 20 \
  --output_dir runs/sdft_lora_mps_20steps
```

Cluster (CUDA, drop laptop-friendly knobs — re-enable vLLM, bf16, longer contexts):
```bash
accelerate launch train_sdft_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --max_steps 2000 \
  --learning_rate 1e-4 \
  --lora_r 32 --lora_alpha 64 \
  --output_dir runs/sdft_lora_7b_cluster
# Then edit build_distil_config to set use_vllm=True, vllm_mode='colocate',
# vllm_gpu_memory_utilization=0.3, bf16=True, max_prompt_length=1024,
# max_completion_length=1024, report_to='wandb', and remove the
# sync_ref_model fp32 caveat. (We did NOT add CLI flags for these to keep the
# laptop entry-point simple; flip them in build_distil_config when porting.)
```

Final LoRA adapter ends up at `<output_dir>/lora_adapter/` via `student_peft.save_pretrained(...)`, with tokenizer alongside. Load later via `PeftModel.from_pretrained(base_model, "<output_dir>/lora_adapter")`.

## Open questions / known caveats
- **Did the script actually run on MPS?** Unknown — blocked at `peft` import. After installing the dep, watch step 1 for: (a) finite loss, (b) non-zero gradient norms on LoRA params, (c) no OOM with student+teacher both fp32 + grad-checkpointing on 24GB.
- **`gradient_checkpointing=True` + PEFT footgun**: HF Trainer normally calls `model.enable_input_require_grads()` when it detects PEFT, but `DistilTrainer` is a custom subclass and we did not grep an explicit handling. If LoRA gradients come back zero on step 1, add `student_peft.enable_input_require_grads()` after `apply_lora()`.
- **Teacher memory on MPS**: a frozen 3B teacher + 3B LoRA student + KV cache + activations on a 24GB Mac is tight. Fallback ladder: drop `max_completion_length` to 128, then 64; last resort, place teacher on CPU (`device_map='cpu'`) since it runs under `torch.no_grad()` at `distil_trainer.py:1629-1633` — slower but functional.
- **`sync_ref_model=True` under LoRA**: the EMA callback iterates `model.parameters()` — under PEFT, that includes both frozen base weights and trainable LoRA deltas. The base weights of student and teacher are byte-identical at init, so EMA over them is a no-op; LoRA params don't exist on the teacher side, so they should be skipped by name. Worth empirically verifying the callback does not throw a key-mismatch error under PEFT. If it does, either disable `sync_ref_model` (and document the fidelity gap vs `main.py`) or filter `named_parameters` to base-only inside the callback.
- **`max_completion_length=256` and demo-following**: SDFT relies on the teacher's demo-conditioned completion distribution being informative on the same prefix the student samples. With completions truncated at 256, longer golden responses get cut — the KL signal may degrade for long-form tool-use rows. For a smoke run this is fine; for a real run consider 512.
- **HF generate path always uses student**: documented in the file's `build_distil_config` comment; if a future maintainer flips `generate_from_teacher=True` expecting teacher-sourced samples, with `use_vllm=False` they will silently get student samples (distil_trainer.py:1226, 1258). Not a bug here, but a maintenance footgun.
- **Holdout JSON only checks `source_rows`, not `source_dataset`**: if `TRAIN_DATA_PATH` is ever repointed to a different 4046-row dataset, the holdout filter would silently mis-drop wrong indices. Reviewer flagged; left as a follow-up.
