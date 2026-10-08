"""Regression tests for make_figures.py's data prep (pandas only; one test needs matplotlib).

Pinned bugs, all visible only in the rendered PNGs, never in a CSV:

1. prep_forgetting_curve took its step-0 anchor from retention.csv without filtering
   eval_set. retention.csv has a tooluse/holdout AND a tooluse/eval_data row per stage-2
   run; dict(zip()) kept the later one, so every *holdout* curve started at its *eval_data*
   stage-1 accuracy (0.59-0.71 instead of 0.31-0.39 at 7B): a fake ~30pp collapse by step 50.
2. The acq50 control was drawn as "SFT" (same colour/marker, pooled into the SFT bar) --
   also when reading tables written before collect_results learned the "sft_acq50" arm.
3. fig_forgetting_bars drew all ~67 lm-eval rows (incl. every mmlu_* subtask) and would
   pool a stage-2 battery into the stage-1 bar (forgetting.csv has no stage column).
4. fig_scale_trend drew a "trend" from a single scale on 3B/7B/14B ticks.

Run: .venv/bin/python -m pytest tests/test_make_figures.py -v
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import make_figures as mf  # noqa: E402


def _rl_row(run, step, value, *, objective="sft", teacher="none", stage=2, eval_set="holdout"):
    return {"run": run, "objective": objective, "teacher": teacher, "stage": stage,
            "checkpoint": f"step{step}", "eval_dataset": "tooluse", "eval_set": eval_set,
            "metric": "accuracy", "value": value}


def test_forgetting_curve_step0_is_stage1_acc_on_the_same_eval_set():
    run = "sft_lora_7b_science_s2_seed42"
    t = {"results_long": pd.DataFrame([_rl_row(run, 50, 0.35), _rl_row(run, 168, 0.36)]),
         # collect_results writes holdout first, then eval_data, for every stage-2 run
         "retention": pd.DataFrame([
             {"arm": run, "eval_set": "tooluse/holdout", "stage1_acc": 0.39},
             {"arm": run, "eval_set": "tooluse/eval_data", "stage1_acc": 0.5876}])}
    steps, accs, arm = mf.prep_forgetting_curve(t)[run]
    assert steps == [0, 50, 168]
    assert accs[0] == pytest.approx(0.39), "step 0 came from the eval_data row"
    assert arm == "sft"

    t_ed = {"results_long": pd.DataFrame([_rl_row(run, 50, 0.63, eval_set="eval_data")]),
            "retention": t["retention"]}
    assert mf.prep_forgetting_curve(t_ed, eval_set="eval_data")[run][1][0] == pytest.approx(0.5876)


def test_acq50_gets_its_own_arm_and_style_even_from_pre_fix_tables():
    acq = "sft_lora_7b_science_s2_acq50_seed42"
    t = {"results_long": pd.DataFrame([_rl_row(acq, 50, 0.31)]),
         "retention": pd.DataFrame([{"arm": acq, "eval_set": "tooluse/holdout", "stage1_acc": 0.33}]),
         # arm_id "sft" for the control = what the pre-fix collect_results wrote
         "continual_metrics": pd.DataFrame([
             {"arm": acq, "arm_id": "sft", "science_acc": 0.574, "bwt_tooluse": -0.01},
             {"arm": "sft_lora_7b_science_s2_seed42", "arm_id": "sft", "science_acc": 0.578,
              "bwt_tooluse": -0.03}])}
    assert mf.prep_forgetting_curve(t)[acq][2] == "sft_acq50"
    assert [p["arm"] for p in mf.prep_tradeoff(t)] == ["sft_acq50", "sft"]
    assert mf.ARM_STYLE["sft_acq50"]["label"] == "SFT-acq50"
    assert mf.ARM_STYLE["sft_acq50"]["color"] != mf.ARM_STYLE["sft"]["color"]


def test_forgetting_bars_headline_tasks_seed_mean_pp_stage1_only():
    def row(run, task, forgetting, arm_id="sft"):
        return {"arm": run, "arm_id": arm_id, "scale": "7b", "task": task, "base": 0.56,
                "base_stderr": 0.0213, "adapted": 0.56 - forgetting, "adapted_stderr": 0.0214,
                "forgetting": forgetting}
    s42, s1234 = "sft_lora_7b_tooluse_s1_seed42", "sft_lora_7b_tooluse_s1_seed1234"
    acq, s2 = "sft_lora_7b_tooluse_s1_acq50_seed42", "sft_lora_7b_science_s2_seed42"
    t = {"forgetting": pd.DataFrame([
            row(s42, "ifeval", 0.098), row(s1234, "ifeval", 0.109),
            row(acq, "ifeval", 0.091),                 # pre-fix arm_id "sft"
            row(s42, "mmlu_anatomy", 0.050),           # subtask: not a headline task
            row(s2, "ifeval", 0.200)]),                # stage-2 battery: not pooled
         "runs_index": pd.DataFrame({"run": [s42, s1234, acq, s2], "stage": [1, 1, 1, 2]})}
    f = mf.prep_forgetting_bars(t).set_index("arm_id")
    assert set(f["task"]) == {"ifeval"}
    assert f.loc["sft", "n_seeds"] == 2
    assert f.loc["sft", "mean_pp"] == pytest.approx(-10.35)       # adapted - base, in pp
    assert sorted(f.loc["sft", "seeds_pp"]) == pytest.approx([-10.9, -9.8])
    assert f.loc["sft_acq50", "mean_pp"] == pytest.approx(-9.1)
    assert f.loc["sft", "se_pp"] == pytest.approx(100 * (0.0213 ** 2 + 0.0214 ** 2) ** 0.5)


def test_scale_trend_needs_two_scales(tmp_path):
    def agg(scale, mean):
        return {"arm": "sft", "scale": scale, "dataset": "tooluse", "stage": 1,
                "eval_set": "tooluse/holdout", "n_seeds": 3, "mean_acc": mean, "std_acc": 0.01}
    one = {"aggregate": pd.DataFrame([agg("7b", 0.38)]), "results_long": pd.DataFrame()}
    assert mf.fig_scale_trend(one, tmp_path) is False
    assert not (tmp_path / "fig_scale_trend.png").exists()

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    two = {"aggregate": pd.DataFrame([agg("3b", 0.30), agg("7b", 0.38)]), "results_long": pd.DataFrame()}
    assert mf.fig_scale_trend(two, tmp_path) is True
    assert (tmp_path / "fig_scale_trend.png").exists()


def test_retention_endpoints_one_panel_per_set_acq50_split():
    def row(run, arm_id, eval_set, s1, s2):
        return {"arm": run, "arm_id": arm_id, "eval_set": eval_set, "stage1_acc": s1, "stage2_acc": s2}
    sft, acq = "sft_lora_7b_science_s2_seed42", "sft_lora_7b_science_s2_acq50_seed42"
    t = {"retention": pd.DataFrame([
        row(sft, "sft", "tooluse/holdout", 0.39, 0.36),
        row(sft, "sft", "tooluse/eval_data", 0.5876, 0.6495),
        row(acq, "sft", "tooluse/holdout", 0.33, 0.32)])}   # pre-fix arm_id for the control
    d = mf.prep_retention_endpoints(t)
    assert d["tooluse/holdout"]["sft"] == [(0.39, 0.36)]
    assert d["tooluse/holdout"]["sft_acq50"] == [(0.33, 0.32)]
    assert d["tooluse/eval_data"] == {"sft": [(0.5876, 0.6495)]}
    assert mf.prep_retention_endpoints({"retention": pd.DataFrame()}) == {}
