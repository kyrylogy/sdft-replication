"""C1 provenance audit: did every lm-eval battery run under identical settings, and
was the SAME metric key read for every run of every task?

The headline general-capability deltas (esp. the 14pp HumanEval SFT-vs-SDFT gap) are
only meaningful if (a) harness version, few-shot counts, and model_args agree across
the base anchor and every arm, and (b) collect_results' metric-key fallback resolved
to the same key everywhere (T11 flagged the fallback as order-dependent).

Reads runs/*/eval/forgetting*/results*.json. No GPU. Exit 1 on any disagreement.

  .venv/bin/python battery_audit.py            # human-readable report
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from collect_results import _pick_lmeval  # replicate EXACTLY the key the tables read

REPO = Path(__file__).resolve().parent


def audit(root="runs"):
    rows = []
    # lm-eval nests results one level below --output_path, in a dir named after model_args
    # (the sanitized adapter path — itself useful provenance for which adapter was loaded).
    for rj in sorted((REPO / root).glob("*/eval/forgetting*/**/results*.json")):
        run = rj.parts[rj.parts.index(root) + 1]
        label = "base" if "forgetting_base" in str(rj) else "arm"
        try:
            d = json.loads(rj.read_text())
        except Exception as e:
            rows.append({"run": run, "label": label, "error": f"{e.__class__.__name__}: {e}"})
            continue
        cfg = d.get("config", {}) or {}
        # n-shot lives at top level in the pinned harness; fall back to per-task configs.
        nshot = d.get("n-shot") or {t: (tc or {}).get("num_fewshot")
                                   for t, tc in (d.get("configs", {}) or {}).items()}
        picked = {}
        for task, tm in (d.get("results", {}) or {}).items():
            if "_" in task and task.startswith("mmlu_"):
                continue  # subtasks inherit mmlu's config; auditing the parent is enough
            mk, _ = _pick_lmeval(tm)
            picked[task] = mk
        rows.append({
            "run": run, "label": label, "path": str(rj.relative_to(REPO)),
            "git_hash": d.get("git_hash"),
            "lm_eval_version": d.get("lm_eval_version") or d.get("version") or cfg.get("lm_eval_version"),
            "model": cfg.get("model"),
            "model_args": cfg.get("model_args"),
            "batch_size": cfg.get("batch_size"),
            "nshot": {k: v for k, v in sorted(nshot.items()) if not k.startswith("mmlu_")},
            "picked_keys": dict(sorted(picked.items())),
            "humaneval_numeric_keys": sorted(k for k, v in (d.get("results", {}).get("humaneval", {}) or {}).items()
                                             if isinstance(v, (int, float))),
        })
    return rows


def main():
    rows = audit()
    if not rows:
        print("[audit] no battery results found under runs/*/eval/forgetting*/")
        sys.exit(1)

    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    print(f"[audit] {len(ok)} battery result files ({sum(1 for r in ok if r['label']=='base')} base anchor)")

    disagreements = 0
    # Fields that must be IDENTICAL across every battery for deltas-vs-base to be valid.
    # model_args is excluded (it names the adapter, so it differs by construction) — but
    # we still show it below so a wrong-adapter battery is visible at a glance.
    for field in ["git_hash", "lm_eval_version", "model", "batch_size", "nshot", "picked_keys"]:
        vals = defaultdict(list)
        for r in ok:
            vals[json.dumps(r[field], sort_keys=True)].append(r["run"])
        if len(vals) == 1:
            print(f"  [OK] {field}: {next(iter(vals))}")
        else:
            disagreements += 1
            print(f"  [DISAGREE] {field}:")
            for v, runs in sorted(vals.items()):
                print(f"      {v}  <-  {', '.join(sorted(set(runs)))}")

    print("\n[audit] per-file model_args (adapter identity — verify each battery loaded its own run's adapter):")
    for r in ok:
        print(f"  {r['run']:42} {r['label']:4} {r['model_args']}")

    print("\n[audit] humaneval numeric keys present (the picked key must be first-preference, not fallback):")
    for r in ok:
        print(f"  {r['run']:42} picked={r['picked_keys'].get('humaneval')} all={r['humaneval_numeric_keys']}")

    for r in errors:
        disagreements += 1
        print(f"  [ERROR] {r['run']}: {r['error']}")

    print(f"\n[audit] {'CLEAN — all batteries comparable' if disagreements == 0 else f'{disagreements} FIELD(S) DISAGREE — fix before headlining battery deltas'}")
    sys.exit(0 if disagreements == 0 else 1)


if __name__ == "__main__":
    main()
