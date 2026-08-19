"""Regression tests for collect_results.py's run discovery + table joins.

These build a tiny synthetic runs/ tree (no GPU, no real model, <1s) reproducing
the on-disk shape eval_runner.py actually writes, then run the real
collect_results.py script against it and assert on the CSVs it emits.

Written after a real bug: base_anchor/ceiling/forgetting_base (--base evals,
which never go through train.py) were invisible to collect_results.py because
load_runs() discovers runs via `runs/*/resolved_config.yaml` (exactly one level
deep) and nothing wrote that file at the run's top level for --base runs. The
result: gap_closed.csv was silently always empty, and forgetting.csv's
base/forgetting columns were silently always null, for a whole 7B tier's worth
of GPU time before anyone noticed. See eval_runner.py's two extra stamp_run()
calls in run_accuracy()/run_forgetting()'s use_base branches for the fix.

Run: .venv/bin/python -m pytest tests/test_collect_results.py -v
(needs only pyyaml + stdlib -- no torch, runs fine locally too)
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _resolved_cfg(*, seed=42, num_fewshot=None):
    """Mirrors what exp_config.stamp_run() actually writes (the fields
    collect_results._meta() reads back out)."""
    return {
        "method": {"objective": "sft", "tuning": "lora", "teacher": "none"},
        "model": {"id": "Qwen/Qwen2.5-7B-Instruct", "scale": "7b"},
        "data": {"dataset": "tooluse", "stage": 1},
        "train": {"seed": seed},
        "eval": {"scorer": "strict", "forgetting": {"num_fewshot": num_fewshot}},
    }


def _acc_result(accuracy, n=100, teacher_ceiling=False):
    return {"accuracy": accuracy, "n_effective": n,
            "wilson95": [max(0.0, accuracy - 0.1), min(1.0, accuracy + 0.1)],
            "teacher_ceiling": teacher_ceiling, "per_sample_scores": [1, 0] * (n // 2)}


def _lmeval_result(task, value, stderr=0.01):
    return {"results": {task: {"acc_norm,none": value, "acc_norm_stderr,none": stderr}}}


@pytest.fixture
def synthetic_runs(tmp_path):
    """One trained arm (sft_lora_7b_tooluse_s1_seed42) + the untagged --base
    anchor dir (base_anchor accuracy, ceiling accuracy, forgetting_base battery)
    it should be compared against -- the exact shape a real 7B tooluse_s1 tier
    produces."""
    root = tmp_path / "runs"

    arm = root / "sft_lora_7b_tooluse_s1_seed42"
    _write_yaml(arm / "resolved_config.yaml", _resolved_cfg(seed=42))
    _write_json(arm / "eval/final/tooluse_holdout/eval_results.json", _acc_result(0.39))
    _write_json(arm / "eval/forgetting/modelhash/results_2026-01-01T00-00-00.json",
                _lmeval_result("hellaswag", 0.80))

    base = root / "sft_lora_7b_tooluse_s1"
    _write_yaml(base / "resolved_config.yaml", _resolved_cfg(seed=42))
    _write_json(base / "eval/base_anchor/tooluse_holdout/eval_results.json", _acc_result(0.27))
    _write_json(base / "eval/ceiling/tooluse_holdout/eval_results.json",
                _acc_result(0.52, teacher_ceiling=True))
    _write_json(base / "eval/forgetting_base/modelhash/results_2026-01-01T00-00-00.json",
                _lmeval_result("hellaswag", 0.75))

    return root


def _run_collect(root, out):
    r = subprocess.run([sys.executable, str(REPO / "collect_results.py"),
                         "--root", str(root), "--out", str(out)],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"collect_results.py failed:\n{r.stdout}\n{r.stderr}"
    return r


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_gap_closed_finds_base_and_ceiling(synthetic_runs, tmp_path):
    out = tmp_path / "analysis"
    _run_collect(synthetic_runs, out)

    rows = _read_csv(out / "gap_closed.csv")
    assert len(rows) == 1, (
        "gap_closed.csv should have one row for the arm's holdout accuracy; "
        "got 0 => base_anchor/ceiling weren't discovered (the top-level "
        "resolved_config.yaml regression)"
    )
    row = rows[0]
    assert row["arm"] == "sft_lora_7b_tooluse_s1_seed42"
    assert float(row["base"]) == pytest.approx(0.27)
    assert float(row["ceiling"]) == pytest.approx(0.52)
    assert float(row["arm_acc"]) == pytest.approx(0.39)
    # (0.39 - 0.27) / (0.52 - 0.27) = 48.0%
    assert float(row["gap_closed_pct"]) == pytest.approx(48.0, abs=0.5)


def test_forgetting_matches_base_by_scale_and_task(synthetic_runs, tmp_path):
    out = tmp_path / "analysis"
    _run_collect(synthetic_runs, out)

    rows = _read_csv(out / "forgetting.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["arm"] == "sft_lora_7b_tooluse_s1_seed42"
    assert row["task"] == "hellaswag"
    assert row["base"] not in ("", None), (
        "base column is empty => forgetting_base wasn't discovered/matched "
        "(same top-level resolved_config.yaml regression as gap_closed)"
    )
    assert float(row["base"]) == pytest.approx(0.75)
    assert float(row["adapted"]) == pytest.approx(0.80)
    assert float(row["forgetting"]) == pytest.approx(0.75 - 0.80)


def test_base_only_runs_are_never_double_counted_as_arms(synthetic_runs, tmp_path):
    """The untagged base dir (sft_lora_7b_tooluse_s1) must not itself show up
    as a trained arm in retention/continual_metrics -- it has no lora_adapter,
    no stage-2, nothing to retain."""
    out = tmp_path / "analysis"
    _run_collect(synthetic_runs, out)

    idx = _read_csv(out / "runs_index.csv")
    names = {r["run"] for r in idx}
    assert "sft_lora_7b_tooluse_s1_seed42" in names
    assert "sft_lora_7b_tooluse_s1" in names  # discovered...
    forg_arms = {r["arm"] for r in _read_csv(out / "forgetting.csv")}
    assert "sft_lora_7b_tooluse_s1" not in forg_arms  # ...but never as an "adapted" row
