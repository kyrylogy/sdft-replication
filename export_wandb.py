#!/usr/bin/env python
"""Export all wandb runs of a project to compact CSVs + a JSON index.

Output is meant to be small enough to paste into an LLM for analysis:
history is downsampled, all-NaN columns dropped, floats rounded.

Usage (on the machine where wandb is logged in):
    python export_wandb.py                     # project from configs/base.yaml default
    python export_wandb.py --entity myuser --samples 500
"""
import argparse
import json
import os

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="sdft-replication")
    ap.add_argument("--entity", default=None, help="wandb entity; defaults to your account")
    ap.add_argument("--samples", type=int, default=300, help="max history rows per run")
    ap.add_argument("--out", default="wandb_export")
    args = ap.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(path)
    os.makedirs(args.out, exist_ok=True)

    index = []
    for run in runs:
        safe = run.name.replace("/", "_")
        summary = {k: v for k, v in run.summary.items()
                   if isinstance(v, (int, float, str, bool))}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
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

    with open(os.path.join(args.out, "runs_index.json"), "w") as f:
        json.dump(index, f, indent=1, default=str)
    print(f"\nWrote {args.out}/ ({len(index)} runs)")


if __name__ == "__main__":
    main()
