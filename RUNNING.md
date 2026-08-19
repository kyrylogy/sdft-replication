# Running on the cluster — the operational guide

How to actually launch, keep alive, monitor, and troubleshoot the experiments on a shared
GPU box. Pairs with [`EXPERIMENTS.md`](EXPERIMENTS.md) (what to run) and [`HARNESS.md`](HARNESS.md)
(how the system works).

---

## 0. One-time setup (per fresh clone)

```bash
cd ~/projects/sdft-replication
# venv on the pinned deps (uv lives in the vllm_venv; installs into a SEPARATE .venv)
source /mnt/data/vllm_venv/bin/activate
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip uninstall --python .venv/bin/python deepspeed          # not needed; crashes w/o CUDA_HOME
uv pip install --python .venv/bin/python \
  "git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd"  # forgetting suite
uv pip install --python .venv/bin/python langdetect immutabledict   # ifeval task needs these; not pulled in above
deactivate
```

Persist these in `~/.bashrc` so every shell (and every tmux pane) has them:
```bash
echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc      # writable model cache (shared /mnt cache is read-only to you)
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc    # from https://wandb.ai/authorize
mkdir -p $HF_HOME
source ~/.bashrc
```

Verify:
```bash
.venv/bin/python -c "import torch,transformers,trl,peft,datasets,yaml,pandas,matplotlib,wandb; print('deps OK')"
```

---

## 1. tmux — survive disconnection

A long run must outlive your SSH session. `tmux` gives you a terminal on the server that
keeps running after you disconnect; you reattach later and everything's still there.

```bash
tmux new -s tier          # create a session named "tier"
#   ... run things inside ...
# detach (leave it running):    Ctrl-b   then   d
tmux ls                   # list sessions
tmux attach -t tier       # reattach
tmux kill-session -t tier # stop it for good
```
Inside tmux: **scroll** with `Ctrl-b` then `[` (arrow keys / PgUp; press `q` to exit scroll).
Split panes (optional): `Ctrl-b "` (horizontal), `Ctrl-b %` (vertical), `Ctrl-b o` to switch.

If `tmux` isn't installed, `screen` works the same way (`screen -S tier` / detach `Ctrl-a d` /
`screen -r tier`), or fall back to `nohup ... &` (§3).

---

## 2. wandb — remote monitoring (no CLI needed)

wandb **is** installed in `.venv`; the `wandb` command just isn't on your PATH. Don't run
`wandb login` (interactive — it blocks under nohup). Instead authenticate with the env var:

```bash
export WANDB_API_KEY=<key>     # already in ~/.bashrc if you did §0
```
That's it — training and eval log automatically (runs grouped by `scale_dataset`). Watch the
dashboard from your laptop at wandb.ai — it survives your SSH dropping. Track `loss`,
`kl_approx` (SDFT: compare shape EMA vs frozen), `entropy`, and the eval-accuracy summaries.

Don't want wandb? Add `--set runtime.report_to=none` (or `tensorboard`) to any run.

---

## 3. Launch the matrix

Everything runs through the resumable driver — one process per scale, gate-safe order,
**skips finished steps** (re-run after a crash and it continues).

```bash
tmux new -s tier
GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh preflight   # dry-run every config, no GPU
GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run         # the 7B spine
# Ctrl-b d to detach.  Do 14B and 3B afterwards (new session or after 7B finishes):
#   GPU=3 SCALE=14b SEEDS="42 1234 2024" ./run_tier.sh run
#   GPU=3 SCALE=3b  SEEDS="42 1234 2024" ./run_tier.sh run
```

nohup alternative (no tmux):
```bash
nohup env GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run > /dev/null 2>&1 &
```
The driver tees its own log to `logs/tier_<scale>_<timestamp>.log` either way.

---

## 4. Monitoring

```bash
tail -f logs/tier_7b_*.log                 # live command-level progress (which step it's on)
.venv/bin/python collect_results.py         # runs/ -> analysis/*.csv  (run anytime)
column -t -s, analysis/runs_index.csv       # coverage: which cells are trained/evaled/forgetting-done
column -t -s, analysis/retention.csv        # the BWT (continual-forgetting) numbers
nvidia-smi                                  # GPU state (you share the box)
```
The driver auto-runs `collect_results` + `make_figures` after the seed-42 shakeout and at the
end. After the shakeout, **eyeball `analysis/retention.csv` before letting the rest grind.**

---

## 5. Shared-GPU etiquette

You can't kill other users' jobs. GPU 3 (~76 GB free) is the workhorse; the driver's
`MIN_FREE_MIB` guard refuses to start if a card is too full. Pick a different card with
`GPU=<id>`. If another user lands on your card mid-run, the vLLM memory cap (§ troubleshooting)
is your protection.

---

## 6. Troubleshooting (errors seen in the wild)

| Symptom | Cause → fix |
|---|---|
| `wandb: command not found` | CLI not on PATH. You don't need it — `export WANDB_API_KEY=<key>`. (Or `.venv/bin/wandb login`.) |
| `PermissionError: /mnt/data/huggingface_cache` | shared cache is read-only to you → `export HF_HOME=$HOME/hf_cache` (run.sh also defaults it). |
| `RepresenterError ... '2.9.0+cu128'` | old yaml-stamp bug → `git pull` (fixed). Instant: `sed -i 's|getattr(importlib.import_module(m), "__version__", "?")|str(&)|' exp_config.py`. |
| `MissingCUDAException: CUDA_HOME does not exist` (deepspeed) | deepspeed unneeded for single-GPU LoRA → `source /mnt/data/vllm_venv/bin/activate; uv pip uninstall --python .venv/bin/python deepspeed; deactivate`. |
| vLLM init: "less than desired" / CUDA OOM | card not empty → lower the reservation: append `--set vllm.gpu_memory_utilization=0.25`, or run vLLM-off with `--set vllm.enabled=false`. |
| `[gate] FAIL-CLOSED: no stage-1 expected accuracy` | you launched stage-2 before stage-1 eval → run the stage-1 accuracy eval first (the driver orders this for you). |
| `git pull` "local changes would be overwritten" | you edited a tracked file (e.g. the sed patch) → `git stash && git pull` (or `git checkout <file> && git pull`). |
| `df: ~/.triton/autotune: No such file` | harmless Triton warning — ignore. |

---

## 7. Quick reference

```bash
# whole 7B tier, detached, monitored via wandb + log
tmux new -s tier
GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run     # Ctrl-b d to detach
# reattach / peek
tmux attach -t tier
tail -f logs/tier_7b_*.log
.venv/bin/python collect_results.py && column -t -s, analysis/retention.csv
```
