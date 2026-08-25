"""Single source of truth for dataset loading + prompt formatting.

Both train.py and evaluate.py (and the stage-2 retention gate) import from here,
so the teacher template, response-only marker, and holdout logic can never drift
between training and evaluation — the "no parallel divergence" rule, enforced by
having exactly one implementation.

Supported datasets: "tooluse" and "science". For each we expose:
  - format_sdft(row)  -> {"prompt", "teacher_prompt"}   (student bare / teacher demo-conditioned)
  - format_sft(row)   -> {"text"}                        (chat-templated, response-only masked downstream)
  - load_train(...)   -> formatted HF Dataset for the chosen objective
  - load_eval(...)    -> list[dict] ready for generation + scoring
"""

from pathlib import Path
from string import Template
import hashlib
import json

from datasets import load_from_disk

_REPO = Path(__file__).resolve().parent

# The teacher demonstration wrapper. Byte-identical to main.py / train_sdft_lora.py
# (leading newline inside the triple-quote is intentional and part of the contract).
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

# Marker the SFT response-only collator masks up to (Qwen2.5 chat template).
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# ---------------------------------------------------------------------------
# Per-dataset path + schema registry. Adding a dataset = one entry here.
# ---------------------------------------------------------------------------
DATASETS = {
    "tooluse": {
        "train": _REPO / "data/tooluse_data/train_data",
        "eval": {
            "eval_data": _REPO / "data/tooluse_data/eval_data",
            "holdout": _REPO / "data/tooluse_data/train_subset_holdout",
        },
        "holdout_indices": _REPO / "data/tooluse_data/train_subset_holdout_indices.json",
        "score": "tooluse",   # action+input matching
    },
    "science": {
        "train": _REPO / "data/science_data/train_data",
        "eval": {
            "eval_data": _REPO / "data/science_data/eval_data",
        },
        "holdout_indices": None,   # science has no train-carve; eval_data is the held-out split
        "score": "science",   # <answer>X</answer> exact match
    },
}


def _require(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {list(DATASETS)}")
    return DATASETS[dataset]


def fingerprint(dataset: str) -> dict:
    """Content hash + byte size of the on-disk arrow files for a dataset's train/eval
    splits. Stamped into each run's record so a silently regenerated dataset is
    detectable across runs (reproducibility guard). Hashing arrow bytes is fast and
    deterministic — no need to materialize rows."""
    if dataset == "joint":  # joint-training ceiling: stamp both constituents
        return {d: fingerprint(d) for d in ("tooluse", "science")}
    spec = _require(dataset)
    roles = [("train", spec["train"])] + [(f"eval:{k}", v) for k, v in spec["eval"].items()]
    out = {}
    for role, base in roles:
        h, total, n = hashlib.sha256(), 0, 0
        for a in sorted(Path(base).glob("*.arrow")):
            b = a.read_bytes()
            h.update(b)
            total += len(b)
            n += 1
        out[role] = {"sha256": h.hexdigest()[:16], "bytes": total, "arrow_files": n}
    return out


# ---------------------------------------------------------------------------
# Row-level formatting (the ONE place prompts are built)
# ---------------------------------------------------------------------------
def format_sdft(row: dict, dataset: str) -> dict:
    """Bare student prompt + demo-conditioned teacher prompt."""
    if dataset == "tooluse":
        orig = row["prompt"]
        demo = "\n".join(row["golden_response"])
        return {
            "prompt": [{"role": "user", "content": orig}],
            "teacher_prompt": [{"role": "user",
                                "content": TEACHER_TEMPLATE.substitute(orig_content=orig, output_text=demo)}],
        }
    if dataset == "science":
        messages = row["messages"]                 # [system, user]
        orig = messages[1]["content"]
        demo = row["output_text"]
        return {
            "prompt": messages,
            "teacher_prompt": [
                messages[0],
                {"role": "user",
                 "content": TEACHER_TEMPLATE.substitute(orig_content=orig, output_text=demo)},
            ],
        }
    raise ValueError(dataset)


def format_sft(row: dict, dataset: str, tokenizer) -> dict:
    """Chat-templated full sequence; ResponseOnlyCollator masks the prompt span."""
    if dataset == "tooluse":
        msgs = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": "\n".join(row["golden_response"])},
        ]
    elif dataset == "science":
        msgs = list(row["messages"]) + [{"role": "assistant", "content": row["output_text"]}]
    else:
        raise ValueError(dataset)
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}


# ---------------------------------------------------------------------------
# Holdout filter (tooluse only)
# ---------------------------------------------------------------------------
def filter_holdout(raw, dataset: str, enabled: bool):
    """Drop the reserved eval indices BEFORE map/shuffle so the index space
    matches the on-disk source dataset. No-op for datasets without a carve."""
    meta_path = _require(dataset)["holdout_indices"]
    if not enabled or meta_path is None:
        if enabled and meta_path is None:
            print(f"[holdout] {dataset} has no holdout manifest; eval uses a separate split. Keeping all rows.")
        return raw
    meta = json.loads(Path(meta_path).read_text())
    holdout = set(meta["indices"])
    assert meta["source_rows"] == len(raw), (
        f"holdout manifest expects {meta['source_rows']} rows but dataset has {len(raw)}; "
        f"refusing to filter against a mismatched index space"
    )
    keep = [i for i in range(len(raw)) if i not in holdout]
    filtered = raw.select(keep)
    print(f"[holdout] dropped {len(holdout)} indices; {len(raw)} -> {len(filtered)} rows")
    return filtered


# ---------------------------------------------------------------------------
# Train loaders
# ---------------------------------------------------------------------------
def load_train(dataset: str, objective: str, seed: int, tokenizer=None,
               holdout_filter: bool = True, max_samples=None):
    """Return a formatted HF Dataset for training.

    objective in {sdft, online_sft} -> {prompt, teacher_prompt} columns.
    objective == sft                 -> {text} column (needs tokenizer).

    dataset == "joint" -> tooluse + science concatenated then shuffled together
    (the joint-training / multi-task ceiling for the continual-learning tables).
    Each constituent is loaded through this same function, so the tooluse holdout
    carve-out applies exactly as it does in the sequential arms.
    """
    if dataset == "joint":
        from datasets import concatenate_datasets
        parts = [load_train(d, objective, seed, tokenizer=tokenizer,
                            holdout_filter=holdout_filter, max_samples=None)
                 for d in ("tooluse", "science")]
        formatted = concatenate_datasets(parts).shuffle(seed=seed)
        print(f"[data] joint: {' + '.join(str(len(p)) for p in parts)} = {len(formatted)} rows")
        if max_samples is not None:
            formatted = formatted.select(range(min(max_samples, len(formatted))))
            print(f"[data] truncated to max_samples={max_samples} -> {len(formatted)} rows")
        return formatted
    spec = _require(dataset)
    raw = load_from_disk(str(spec["train"]))
    print(f"[data] {dataset}/train: {len(raw)} raw rows")
    raw = filter_holdout(raw, dataset, holdout_filter)

    if objective == "sft":
        if tokenizer is None:
            raise ValueError("sft formatting needs a tokenizer")
        formatted = raw.map(lambda r: format_sft(r, dataset, tokenizer),
                            remove_columns=raw.column_names)
    elif objective in ("sdft", "online_sft"):
        formatted = raw.map(lambda r: format_sdft(r, dataset),
                            remove_columns=raw.column_names)
    else:
        raise ValueError(f"unknown objective {objective!r}")

    formatted = formatted.shuffle(seed=seed)
    if max_samples is not None:
        n = min(max_samples, len(formatted))
        formatted = formatted.select(range(n))
        print(f"[data] truncated to max_samples={max_samples} -> {len(formatted)} rows")
    return formatted


# ---------------------------------------------------------------------------
# Eval loader
# ---------------------------------------------------------------------------
def _format_demo_from_row(row):
    """Matched per-row demo string for teacher-ceiling. Prefer real golden_response
    (with Thought), else synthesize Action+Input from golden_answer."""
    gr = row.get("golden_response")
    if gr:
        return "\n".join(gr), "golden_response"
    parts = [f"Action: {s['Action']}\nAction Input: {s['Action_Input']}" for s in row["golden_answer"]]
    return "\n".join(parts), "synthesized_from_golden_answer"


def load_eval(dataset: str, set_name: str, tokenizer, teacher_ceiling: bool = False, max_samples=None):
    """Load an eval split as a list of dicts with a chat-templated 'text' prompt
    plus the fields the scorer needs. Returns (rows, demo_source)."""
    spec = _require(dataset)
    if set_name not in spec["eval"]:
        raise ValueError(f"{dataset} has no eval set {set_name!r}; known: {list(spec['eval'])}")
    data = load_from_disk(str(spec["eval"][set_name])).to_list()
    demo_source = None

    for ex in data:
        if dataset == "tooluse":
            if teacher_ceiling:
                demo_text, src = _format_demo_from_row(ex)
                demo_source = demo_source or src
                content = TEACHER_TEMPLATE.substitute(orig_content=ex["prompt"], output_text=demo_text)
            else:
                content = ex["prompt"]
            ex["text"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        elif dataset == "science":
            messages = ex["prompt"]   # [system, user]
            if teacher_ceiling:
                raise ValueError("teacher_ceiling is not defined for science eval_data (no golden demo column)")
            ex["text"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError(dataset)

    if max_samples is not None:
        data = data[:max_samples]
    return data, demo_source
