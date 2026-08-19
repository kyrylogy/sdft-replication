"""Regression tests for eval_runner.py's --base (anchor) code paths.

No GPU, no real model: _eval_one/_wandb_log_eval/subprocess.run/D.fingerprint
are all monkeypatched out. These import eval_runner.py, which imports torch
at module level, so this file needs the project's real environment
(.venv/bin/python -- not a bare local interpreter):

  .venv/bin/python -m pytest tests/test_eval_runner.py -v

Two real bugs pinned here, both found only after burning real GPU hours:

1. run_forgetting() shelled out to bare "python" instead of sys.executable,
   so the lm_eval subprocess grabbed whatever venv happened to be active in
   the caller's shell -- not the one lm_eval was installed into. Every
   forgetting-battery eval failed with "No module named lm_eval" despite
   `.venv/bin/python -c "import lm_eval"` succeeding directly.

2. --base evals (base_anchor/ceiling/forgetting_base) never go through
   train.py, so nothing stamped the run's top-level resolved_config.yaml.
   collect_results.py discovers runs via runs/*/resolved_config.yaml (one
   level deep) and silently never saw these runs -- gap_closed.csv and
   forgetting.csv's base/forgetting columns were empty for a whole tier.
   See tests/test_collect_results.py for the collect_results.py side of this.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

eval_runner = pytest.importorskip(
    "eval_runner", reason="needs the project .venv (torch, etc.) -- run via .venv/bin/python"
)


def _minimal_cfg(tmp_path, *, teacher_ceiling=False, forgetting_enabled=True):
    return {
        "model": {"id": "fake/model", "dtype": "bfloat16", "scale": "7b"},
        "method": {"objective": "sft", "tuning": "lora", "teacher": "none"},
        "data": {"dataset": "tooluse", "stage": 1},
        "train": {"seed": 42},
        "eval": {
            "sets": ["holdout"], "scorer": "strict", "engine": "hf",
            "max_new_tokens": 64, "temperature": 0.0,
            "teacher_ceiling": teacher_ceiling,
            "forgetting": {"enabled": forgetting_enabled, "tasks": ["hellaswag"],
                            "num_fewshot": None, "batch_size": 8},
        },
        "_derived": {"output_dir": str(tmp_path / "sft_lora_7b_tooluse_s1")},
    }


class _FakeCompletedProcess:
    returncode = 0


def test_forgetting_subprocess_uses_sys_executable(tmp_path, monkeypatch):
    cfg = _minimal_cfg(tmp_path)
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeCompletedProcess()

    monkeypatch.setattr(eval_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(eval_runner, "stamp_run", lambda *a, **k: None)

    eval_runner.run_forgetting(cfg, use_base=True)

    assert "cmd" in captured, "subprocess.run was never called"
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][0] != "python", (
        "regression: bare 'python' resolves via PATH, not the venv eval_runner.py "
        "itself is running under"
    )


def test_forgetting_base_stamps_run_dir_for_discovery(tmp_path, monkeypatch):
    cfg = _minimal_cfg(tmp_path)
    run_dir = Path(cfg["_derived"]["output_dir"])
    stamped = []

    monkeypatch.setattr(eval_runner.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
    monkeypatch.setattr(eval_runner, "stamp_run", lambda cfg, d, **k: stamped.append(Path(d)))

    eval_runner.run_forgetting(cfg, use_base=True)

    assert run_dir in stamped, (
        f"top-level {run_dir} never stamped -- collect_results.py's "
        "runs/*/resolved_config.yaml discovery would never see this run "
        f"(only stamped: {stamped})"
    )


def test_forgetting_arm_run_does_not_need_extra_stamp(tmp_path, monkeypatch):
    """Non-base forgetting runs already get their top-level stamp from
    train.py -- run_forgetting should stamp its own eval subdir only, not
    duplicate a top-level one (that'd just be train.py's file, harmlessly
    overwritten, but asserting the code path stays narrow)."""
    cfg = _minimal_cfg(tmp_path)
    run_dir = Path(cfg["_derived"]["output_dir"])
    stamped = []

    monkeypatch.setattr(eval_runner.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
    monkeypatch.setattr(eval_runner, "stamp_run", lambda cfg, d, **k: stamped.append(Path(d)))

    eval_runner.run_forgetting(cfg, adapter_override=str(tmp_path / "adapter"), use_base=False)

    assert run_dir not in stamped
    assert stamped == [run_dir / "eval" / "forgetting"]


@pytest.mark.parametrize("teacher_ceiling,label", [(False, "base_anchor"), (True, "ceiling")])
def test_accuracy_base_stamps_run_dir_for_discovery(tmp_path, monkeypatch, teacher_ceiling, label):
    cfg = _minimal_cfg(tmp_path, teacher_ceiling=teacher_ceiling)
    run_dir = Path(cfg["_derived"]["output_dir"])
    stamped = []

    monkeypatch.setattr(eval_runner, "stamp_run", lambda cfg, d, **k: stamped.append(Path(d)))
    monkeypatch.setattr(eval_runner.D, "fingerprint", lambda *a, **k: "fake-fingerprint")
    monkeypatch.setattr(eval_runner, "_eval_one", lambda *a, **k: {"accuracy": 0.5})
    monkeypatch.setattr(eval_runner, "_wandb_log_eval", lambda *a, **k: None)

    res = eval_runner.run_accuracy(cfg, use_base=True)

    assert label in res
    assert run_dir in stamped, (
        f"top-level {run_dir} never stamped for the {label} eval -- "
        f"collect_results.py would never discover it (only stamped: {stamped})"
    )
    assert run_dir / "eval" / label in stamped  # its own subdir stamp still happens too
