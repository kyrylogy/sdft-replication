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


def _resolved_cfg(*, seed=42, num_fewshot=None, objective="sft", teacher="none",
                  dataset="tooluse", stage=1, init_adapter=None):
    """Mirrors what exp_config.stamp_run() actually writes (the fields
    collect_results._meta() reads back out)."""
    return {
        "method": {"objective": objective, "tuning": "lora", "teacher": teacher},
        "model": {"id": "Qwen/Qwen2.5-7B-Instruct", "scale": "7b"},
        "data": {"dataset": dataset, "stage": stage, "init_adapter": init_adapter},
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


# ---------------------------------------------------------------------------
# The acq50 control (an SFT stage-1 checkpoint promoted to runs/*acq50*, then
# continued on stage 2) was arm "sft": aggregate.csv pooled it into the SFT seed
# mean (n_seeds=6), significance.csv skipped every SFT-vs-acq50 pair as "same
# arm", and method_effect.csv was right only because "acq50" < "seed" sorts the
# control first, so the full-SFT run overwrote it in a dict.
# ---------------------------------------------------------------------------
def _chain(root, s1, s2, *, seed, s1_acc, s2_acc, objective="sft", teacher="none"):
    """A stage-1 run + the stage-2 run continued from it, laid out like the 7B tier."""
    _write_yaml(root / s1 / "resolved_config.yaml",
                _resolved_cfg(seed=seed, objective=objective, teacher=teacher))
    _write_json(root / s1 / "eval/final/tooluse_holdout/eval_results.json", _acc_result(s1_acc))
    _write_yaml(root / s2 / "resolved_config.yaml",
                _resolved_cfg(seed=seed, objective=objective, teacher=teacher, dataset="science",
                              stage=2, init_adapter=f"runs/{s1}/lora_adapter"))
    _write_json(root / s2 / "eval/final/tooluse_holdout/eval_results.json", _acc_result(s2_acc))
    _write_json(root / s2 / "eval/final/science_eval_data/eval_results.json", _acc_result(0.55))


@pytest.fixture(params=["acq50", "xacq50"], ids=["real-name", "sorts-after-sft"])
def acq50_runs(tmp_path, request):
    """Seed 42 of the three chains: SFT 0.39 -> 0.36, SFT-acq50 0.33 -> 0.32, SDFT-EMA
    0.31 -> 0.30. "xacq50" makes the control sort AFTER full SFT, which is what flipped
    method_effect.csv under the pre-fix code."""
    root, tag = tmp_path / "runs", request.param
    _chain(root, "sft_lora_7b_tooluse_s1_seed42", "sft_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.39, s2_acc=0.36)
    _chain(root, f"sft_lora_7b_tooluse_s1_{tag}_seed42", f"sft_lora_7b_science_s2_{tag}_seed42",
           seed=42, s1_acc=0.33, s2_acc=0.32)
    _chain(root, "sdft_ema_lora_7b_tooluse_s1_seed42", "sdft_ema_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.31, s2_acc=0.30, objective="sdft", teacher="ema")
    return root, tag


def test_acq50_is_its_own_arm_in_every_table(acq50_runs, tmp_path):
    root, tag = acq50_runs
    out = tmp_path / "analysis"
    _run_collect(root, out)

    agg = {(r["arm"], r["stage"], r["eval_set"]): r for r in _read_csv(out / "aggregate.csv")}
    s1 = agg[("sft", "1", "tooluse/holdout")]
    assert s1["n_seeds"] == "1", "acq50 was pooled into the SFT seed mean"
    assert float(s1["mean_acc"]) == pytest.approx(0.39)
    assert float(agg[("sft_acq50", "1", "tooluse/holdout")]["mean_acc"]) == pytest.approx(0.33)
    assert float(agg[("sft", "2", "tooluse/holdout")]["mean_acc"]) == pytest.approx(0.36)
    assert float(agg[("sft_acq50", "2", "tooluse/holdout")]["mean_acc"]) == pytest.approx(0.32)

    for table in ("retention.csv", "continual_metrics.csv"):
        ids = {r["arm"]: r["arm_id"] for r in _read_csv(out / table)}
        assert ids["sft_lora_7b_science_s2_seed42"] == "sft"
        assert ids[f"sft_lora_7b_science_s2_{tag}_seed42"] == "sft_acq50", table

    pairs = {frozenset((r["arm_id_a"], r["arm_id_b"])) for r in _read_csv(out / "significance.csv")}
    assert frozenset(("sft", "sft_acq50")) in pairs, "SFT-vs-acq50 pairs were skipped as 'same arm'"


def test_method_effect_uses_full_sft_whatever_the_dir_order(acq50_runs, tmp_path):
    root, _ = acq50_runs
    out = tmp_path / "analysis"
    _run_collect(root, out)

    me = {(r["protocol"], r["seed"]): r for r in _read_csv(out / "method_effect.csv")}
    assert float(me[("acquisition", "42")]["sft"]) == pytest.approx(0.39)        # not acq50's 0.33
    assert float(me[("retention_bwt", "42")]["sft"]) == pytest.approx(-0.03)     # not acq50's -0.01
    assert float(me[("retention_bwt", "42")]["diff"]) == pytest.approx(-0.01 - -0.03)


def test_joint_ceiling_is_its_own_arm_not_a_second_sft_seed(tmp_path):
    """The joint-training ceiling (SFT on tooluse+science, data.dataset joint, seed 42) is also
    evaluated on tooluse/holdout. As arm "sft" it was a second SFT run in method_effect's seed-42
    cell (kept out only by "joint" < "tooluse" sort order before the duplicate guard, dropped
    as a duplicate after it) and an "SFT" partner in significance.csv."""
    root = tmp_path / "runs"
    _chain(root, "sft_lora_7b_tooluse_s1_seed42", "sft_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.39, s2_acc=0.36)
    _chain(root, "sdft_ema_lora_7b_tooluse_s1_seed42", "sdft_ema_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.31, s2_acc=0.30, objective="sdft", teacher="ema")
    joint = root / "sft_lora_7b_joint_s1_seed42"
    _write_yaml(joint / "resolved_config.yaml", _resolved_cfg(seed=42, dataset="joint"))
    _write_json(joint / "eval/final/tooluse_holdout/eval_results.json", _acc_result(0.42))
    out = tmp_path / "analysis"
    _run_collect(root, out)

    me = {(r["protocol"], r["seed"]): r for r in _read_csv(out / "method_effect.csv")}
    assert float(me[("acquisition", "42")]["sft"]) == pytest.approx(0.39)
    assert not (out / "pairing_issues.csv").exists()
    agg = {(r["arm"], r["dataset"]) for r in _read_csv(out / "aggregate.csv")}
    assert ("sft_joint", "joint") in agg and ("sft", "joint") not in agg
    sig_arms = {r["arm_id_a"] for r in _read_csv(out / "significance.csv")} | \
               {r["arm_id_b"] for r in _read_csv(out / "significance.csv")}
    assert "sft_joint" in sig_arms


def test_method_effect_excludes_cells_with_two_runs_of_one_arm(tmp_path):
    """E.g. a re-run pair kept side by side: excluded + logged, never resolved by dict order."""
    root = tmp_path / "runs"
    _chain(root, "sft_lora_7b_tooluse_s1_seed42", "sft_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.39, s2_acc=0.36)
    _chain(root, "sft_lora_7b_tooluse_s1_rerun_seed42", "sft_lora_7b_science_s2_rerun_seed42",
           seed=42, s1_acc=0.35, s2_acc=0.30)
    _chain(root, "sdft_ema_lora_7b_tooluse_s1_seed42", "sdft_ema_lora_7b_science_s2_seed42",
           seed=42, s1_acc=0.31, s2_acc=0.30, objective="sdft", teacher="ema")
    out = tmp_path / "analysis"
    _run_collect(root, out)

    assert not any(r["seed"] == "42" for r in _read_csv(out / "method_effect.csv"))
    issues = [r for r in _read_csv(out / "pairing_issues.csv") if r["kind"] == "method_effect"]
    assert {r["vs"] for r in issues} == {"acquisition", "retention_bwt"}
    strict = subprocess.run([sys.executable, str(REPO / "collect_results.py"), "--root", str(root),
                             "--out", str(tmp_path / "strict"), "--strict"],
                            capture_output=True, text=True, cwd=REPO)
    assert strict.returncode != 0


def test_humaneval_pass_at_1_is_picked_by_name_not_json_order(tmp_path):
    """lm-eval 0.4.x reports HumanEval as "pass@1,create_test" (filter create_test). That key
    was not in LMEVAL_PRIMARY, so _pick_lmeval fell through to "first numeric non-stderr key
    in JSON order": right for the real k=[1] files, wrong once any other key precedes it."""
    def he(p1, p10):
        return {"results": {"humaneval": {"alias": "humaneval", "pass@10,create_test": p10,
                                          "pass@1,create_test": p1,
                                          "pass@1_stderr,create_test": 0.038}}}
    root = tmp_path / "runs"
    arm = root / "sft_lora_7b_tooluse_s1_seed42"
    _write_yaml(arm / "resolved_config.yaml", _resolved_cfg(seed=42))
    _write_json(arm / "eval/forgetting/modelhash/results_2026-01-01T00-00-00.json", he(0.57, 0.80))
    base = root / "sft_lora_7b_tooluse_s1"
    _write_yaml(base / "resolved_config.yaml", _resolved_cfg(seed=42))
    _write_json(base / "eval/forgetting_base/modelhash/results_2026-01-01T00-00-00.json", he(0.64, 0.85))
    out = tmp_path / "analysis"
    _run_collect(root, out)

    (row,) = _read_csv(out / "forgetting.csv")
    assert float(row["adapted"]) == pytest.approx(0.57)
    assert float(row["base"]) == pytest.approx(0.64)
    assert float(row["adapted_stderr"]) == pytest.approx(0.038)
