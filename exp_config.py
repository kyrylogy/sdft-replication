"""Config backbone: load -> merge -> validate -> resolve -> stamp.

A run is *fully* described by one resolved config. `load_config` merges, in order:
    configs/base.yaml  ->  each file in `extends:`  ->  the experiment file  ->  CLI --set overrides
then validates cross-field rules and derives output paths. `stamp_run` writes the
fully-resolved config plus provenance (git sha, GPU, host, lib versions, timestamp)
into <output_dir>/resolved_config.yaml — the permanent, diffable record of exactly
what ran. Nothing in train.py/evaluate.py reads raw YAML; they consume the resolved dict.
"""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent
CONFIGS_DIR = _REPO / "configs"
BASE_CONFIG = CONFIGS_DIR / "base.yaml"


# ---------------------------------------------------------------------------
# Merge + dotted overrides
# ---------------------------------------------------------------------------
def deep_merge(a: dict, b: dict) -> dict:
    """Recursive dict merge (b wins). Non-dict values (incl. lists) replace wholesale."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _resolve_extends(path: Path, _seen: set) -> dict:
    """Merge a config's `extends:` parents (depth-first, left-to-right) under itself."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular extends at {path}")
    _seen.add(path)
    cfg = _read_yaml(path)
    parents = cfg.pop("extends", []) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for p in parents:
        cand = (path.parent / p)
        if not cand.exists():
            cand = CONFIGS_DIR / p
        merged = deep_merge(merged, _resolve_extends(cand, set(_seen)))
    return deep_merge(merged, cfg)


def _coerce(value: str) -> Any:
    """Parse a --set value string with YAML so 1e-4->float, true->bool, [a,b]->list."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """cfg['train']['learning_rate'] = value  from  'train.learning_rate'."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {dotted}: '{k}' is not a section")
    node[keys[-1]] = value


def parse_overrides(pairs: list[str]) -> dict:
    """['train.learning_rate=1e-4', 'runtime.gpu=0'] -> nested dict of coerced values."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        set_dotted(out, key.strip(), _coerce(raw.strip()))
    return out


# ---------------------------------------------------------------------------
# Load + validate + derive
# ---------------------------------------------------------------------------
def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = deep_merge(_read_yaml(BASE_CONFIG), _resolve_extends(path, set()))
    cfg = deep_merge(cfg, parse_overrides(overrides or []))
    _validate(cfg)
    _derive(cfg)
    return cfg


def _need(cfg: dict, dotted: str):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or node.get(k) is None:
            raise ValueError(f"required config field missing: {dotted}")
        node = node[k]
    return node


def _validate(cfg: dict) -> None:
    errors, warnings = [], []

    def req(dotted):
        try:
            _need(cfg, dotted)
        except ValueError as e:
            errors.append(str(e))

    for f in ["experiment.name", "model.id", "model.scale",
              "method.objective", "method.tuning", "method.teacher",
              "train.learning_rate", "data.dataset", "data.stage"]:
        req(f)

    objective = cfg.get("method", {}).get("objective")
    tuning = cfg.get("method", {}).get("tuning")
    teacher = cfg.get("method", {}).get("teacher")
    scale = cfg.get("model", {}).get("scale")
    stage = cfg.get("data", {}).get("stage")

    if objective not in {None, "sft", "sdft", "online_sft"}:
        errors.append(f"method.objective must be sft|sdft|online_sft, got {objective!r}")
    if tuning not in {None, "lora", "full"}:
        errors.append(f"method.tuning must be lora|full, got {tuning!r}")
    if teacher not in {None, "ema", "frozen", "none"}:
        errors.append(f"method.teacher must be ema|frozen|none, got {teacher!r}")

    # The 14B memory guardrails key off the free-text scale label; make sure it matches
    # the actual model id so a stale label can't silently disable them.
    mid = (cfg.get("model", {}).get("id") or "").lower()
    if scale and mid and str(scale).lower() not in mid:
        warnings.append(f"model.scale={scale!r} not found in model.id={mid!r} — the scale label "
                        f"drives the 14B/memory guardrails; make sure it matches the id.")

    # Objective <-> teacher coherence.
    if objective == "sft" and teacher != "none":
        errors.append("method.teacher must be 'none' when objective=sft")
    if objective in {"sdft", "online_sft"} and teacher in {None, "none"}:
        errors.append(f"method.teacher must be ema|frozen when objective={objective}")

    # online_sft can only sample from the teacher via vLLM (HF generate always uses student).
    if objective == "online_sft" and not cfg.get("vllm", {}).get("enabled"):
        errors.append("objective=online_sft requires vllm.enabled=true (HF generate always samples the student)")

    # Sequential continuation.
    if stage not in {None, 1, 2}:
        errors.append(f"data.stage must be 1 or 2, got {stage!r}")
    if stage == 2 and not cfg.get("data", {}).get("init_adapter"):
        errors.append("data.stage=2 requires data.init_adapter (path to the stage-1 adapter to continue)")
    if stage == 2 and tuning == "full" and not cfg.get("data", {}).get("init_checkpoint"):
        errors.append("data.stage=2 with tuning=full requires data.init_checkpoint (stage-1 full weights)")

    # Memory guardrails (reviewer's teacher-tax + 14B math).
    if scale == "14b" and tuning == "full":
        errors.append("full fine-tune at 14B needs ~250GB with the frozen teacher — refusing. "
                      "Use tuning=lora at 14B, or override model.scale if your budget really allows it.")
    if scale == "14b" and tuning == "lora" and cfg.get("vllm", {}).get("enabled"):
        warnings.append("14B LoRA + colocated vLLM holds student base + teacher base + a vLLM weight copy "
                        "(~3x28GB) and will likely OOM one 80GB card. Supported fix: vllm.enabled=false "
                        "(HF-generate rollouts fit ~60GB on one 80GB card), or place the teacher on a 2nd card.")
    if cfg.get("vllm", {}).get("share_base"):
        warnings.append("vllm.share_base is NOT implemented (sharing one base across accelerate-prepared "
                        "student+teacher needs a DistilTrainer patch); it is a no-op. At 14B use vllm.enabled=false.")

    # Eval engine vs adapter.
    if cfg.get("eval", {}).get("engine") == "vllm" and tuning == "lora":
        warnings.append("eval.engine=vllm cannot load a LoRA adapter; use engine=hf or merge first.")

    for w in warnings:
        print(f"[config][warn] {w}")
    if errors:
        raise SystemExit("[config] invalid:\n  - " + "\n  - ".join(errors))


def _derive(cfg: dict) -> None:
    name = cfg["experiment"]["name"]
    tag = cfg["experiment"].get("tag")
    stem = f"{name}_{tag}" if tag else name
    output_root = cfg.get("runtime", {}).get("output_root", "runs")
    output_dir = str(_REPO / output_root / stem)
    cfg.setdefault("_derived", {})
    cfg["_derived"]["run_stem"] = stem
    cfg["_derived"]["output_dir"] = output_dir
    cfg["_derived"]["run_name"] = stem
    cfg["_derived"]["repo"] = str(_REPO)


# ---------------------------------------------------------------------------
# Provenance + stamping
# ---------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def provenance() -> dict:
    import importlib
    libs = {}
    for m in ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm"]:
        try:
            libs[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:
            libs[m] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "libs": libs,
    }


def stamp_run(cfg: dict, output_dir: str | Path, extra: dict | None = None) -> Path:
    """Write the fully-resolved config + provenance into the run dir. The record."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {k: v for k, v in cfg.items() if k != "_derived"}
    record["_derived"] = cfg.get("_derived", {})
    record["_provenance"] = provenance()
    if extra:
        record["_run"] = extra
    out = output_dir / "resolved_config.yaml"
    out.write_text(yaml.safe_dump(record, sort_keys=False, default_flow_style=False))
    print(f"[stamp] wrote run record -> {out}")
    return out


def dump(cfg: dict) -> str:
    return yaml.safe_dump({k: v for k, v in cfg.items() if k != "_derived"},
                          sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    # `python exp_config.py <config> [--set k=v ...]` -> print resolved config (no run).
    import argparse
    ap = argparse.ArgumentParser(description="Resolve + validate a config, print it.")
    ap.add_argument("config")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    a = ap.parse_args()
    c = load_config(a.config, a.overrides)
    print(dump(c))
    print(f"# output_dir -> {c['_derived']['output_dir']}")
