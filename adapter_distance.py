"""Distance-from-base of a LoRA adapter: ||ΔW||_F over the merged update.

Why this exists: acquisition-matching two arms on *accuracy* does not match them
on how far the adapter actually moved the model. RL's Razor predicts forgetting
from distance-to-base, not from task accuracy — so an acquisition-matched SFT
checkpoint that forgets less might simply have moved less. Recording ||ΔW||_F
for every stage-1 endpoint lets you plot forgetting against distance and say
which axis actually predicts it.

ΔW = (alpha / r) * B @ A  per adapted module. We never form B@A (it is
[out x in], up to 18944x3584 at 7B). Instead, since
    ||BA||_F^2 = tr((BA)^T (BA)) = tr(A^T B^T B A) = tr((B^T B)(A A^T))
and both B^T B and A A^T are [r x r] and symmetric,
    ||BA||_F^2 = sum_ij (B^T B)_ij * (A A^T)_ij
which is exact and costs two [r x r] gram matrices per module.

Usage:
    python adapter_distance.py runs/sft_lora_7b_tooluse_s1_seed42/lora_adapter [...]
    python adapter_distance.py --glob 'runs/*_7b_tooluse_s1_seed*/lora_adapter'
    python adapter_distance.py --glob '...' --csv analysis/adapter_distance.csv
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

# base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight -> (prefix, 'A')
_LORA_RE = re.compile(r"^(?P<mod>.+)\.lora_(?P<which>[AB])\.(?:weight|default\.weight)$")


def _scale(adapter_dir: Path) -> tuple[float, int, int]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        return alpha / (r ** 0.5), r, int(alpha)
    return alpha / r, r, int(alpha)


def adapter_norm(adapter_dir: str | Path):
    """Returns (total_frobenius, {module_suffix: frobenius}, meta)."""
    d = Path(adapter_dir)
    sd = load_file(d / "adapter_model.safetensors")
    scale, r, alpha = _scale(d)

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in sd.items():
        m = _LORA_RE.match(k)
        if m:
            pairs[m.group("mod")][m.group("which")] = v

    per_module: dict[str, float] = {}
    total_sq = 0.0
    for mod, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].to(torch.float64)   # [r, in]
        B = ab["B"].to(torch.float64)   # [out, r]
        gA = A @ A.T                    # [r, r]
        gB = B.T @ B                    # [r, r]
        sq = float((gB * gA).sum()) * (scale ** 2)
        sq = max(sq, 0.0)               # guard fp noise on a zero-init B
        per_module[mod] = sq ** 0.5
        total_sq += sq

    meta = {"r": r, "alpha": alpha, "scale": scale, "n_modules": len(per_module)}
    return total_sq ** 0.5, per_module, meta


def _group(mod: str) -> str:
    """q_proj / up_proj / ... from a full module path."""
    tail = mod.split(".")[-1]
    return tail if tail.endswith("_proj") else mod


def main():
    ap = argparse.ArgumentParser(description="Frobenius norm of merged LoRA deltas")
    ap.add_argument("adapters", nargs="*", help="adapter directories")
    ap.add_argument("--glob", help="glob pattern for adapter dirs")
    ap.add_argument("--csv", help="write a per-adapter CSV here")
    ap.add_argument("--by-module", action="store_true", help="also print per-projection breakdown")
    a = ap.parse_args()

    dirs = list(a.adapters)
    if a.glob:
        dirs += sorted(globmod.glob(a.glob))
    dirs = [d for d in dirs if (Path(d) / "adapter_model.safetensors").exists()]
    if not dirs:
        raise SystemExit("no adapter dirs found (need adapter_model.safetensors)")

    rows, bad = [], []
    for d in dirs:
        try:
            total, per_mod, meta = adapter_norm(d)
        except Exception as e:
            # a truncated/partial copy must not abort the sweep — report it and move on
            bad.append((d, f"{type(e).__name__}: {e}"))
            print(f"{'SKIP':>12}   {d}   ({type(e).__name__})")
            continue
        rows.append({"adapter": d, "frobenius": round(total, 4), **meta})
        print(f"{total:12.4f}   {d}   (r={meta['r']} alpha={meta['alpha']} "
              f"scale={meta['scale']:.2f} modules={meta['n_modules']})")
        if a.by_module:
            agg: dict[str, float] = defaultdict(float)
            for mod, v in per_mod.items():
                agg[_group(mod)] += v ** 2
            for g, sq in sorted(agg.items(), key=lambda kv: -kv[1]):
                print(f"        {g:12} {sq ** 0.5:10.4f}")

    if bad:
        print(f"\n[warn] {len(bad)} adapter(s) unreadable — likely an incomplete copy:")
        for d, why in bad:
            print(f"   {d}\n      {why}")

    if a.csv and rows:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[write] {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
