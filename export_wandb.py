#!/usr/bin/env python
"""Export all wandb runs of a project to compact CSVs + a JSON index.

Output is meant to be small enough to paste into an LLM for analysis:
history is downsampled, all-NaN columns dropped, floats rounded. Machine-local
config fields (absolute paths, host names) are dropped from runs_index.json.

Usage (on the machine where wandb is logged in):
    python export_wandb.py                     # project from configs/base.yaml default
    python export_wandb.py --entity myuser --samples 500
    python export_wandb.py --clean-index       # re-clean an existing runs_index.json (no wandb login)
"""
import argparse
import json
import os
import sys

# The repo root contains a local `wandb/` log directory that shadows the
# installed wandb package — drop the script's own dir from sys.path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

# Config keys that only describe the training machine: output_dir/logging_dir are absolute
# paths (logging_dir also embeds the host name), config_source is lm-eval's absolute task path.
MACHINE_KEYS = {"output_dir", "logging_dir", "config_source"}


def clean(obj):
    """Drop MACHINE_KEYS at any depth."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in MACHINE_KEYS}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="sdft-replication")
    ap.add_argument("--entity", default=None, help="wandb entity; defaults to your account")
    ap.add_argument("--samples", type=int, default=300, help="max history rows per run")
    ap.add_argument("--out", default="wandb_export")
    ap.add_argument("--clean-index", action="store_true",
                    help="only drop machine-local fields from an existing <out>/runs_index.json")
    args = ap.parse_args()

    index_path = os.path.join(args.out, "runs_index.json")
    if args.clean_index:
        with open(index_path) as f:
            index = clean(json.load(f))
        with open(index_path, "w", newline="\n") as f:
            json.dump(index, f, indent=1, default=str)
        print(f"Cleaned {index_path} ({len(index)} runs)")
        return

    import pandas  # noqa: F401  (run.history(pandas=True) silently returns a list without it)
    import wandb

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(path)
    os.makedirs(args.out, exist_ok=True)

    index = []
    for run in runs:
        safe = run.name.replace("/", "_")
        summary = {k: v for k, v in run.summary.items()
                   if isinstance(v, (int, float, str, bool))}
        config = clean({k: v for k, v in run.config.items() if not k.startswith("_")})
        index.append({"name": run.name, "id": run.id, "state": run.state,
                      "created": str(run.created_at),
                      "summary": summary, "config": config})

        hist = run.history(samples=args.samples, pandas=True)
        n_rows = 0 if hist is None else len(hist)
        if n_rows:
            hist = hist.drop(columns=[c for c in hist.columns if hist[c].isna().all()])
            hist = hist.round(5)
            hist.to_csv(os.path.join(args.out, f"{safe}_{run.id}.csv"), index=False)
        print(f"{run.name} ({run.id}): {run.state}, {n_rows} history rows")

    with open(index_path, "w", newline="\n") as f:
        json.dump(index, f, indent=1, default=str)
    print(f"\nWrote {args.out}/ ({len(index)} runs)")


if __name__ == "__main__":
    main()
