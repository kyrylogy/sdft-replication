"""Tests for make_summary.py: analysis/*.csv -> analysis/SUMMARY.md.

Builds a two-arm, two-stage synthetic runs/ tree, runs collect_results.py and make_summary.py on it,
and checks the cells the thesis reads off the summary: exact item-count accuracies (not the 4-dp
CSV values), BWT in pp, pooled forgot/gained counts, and the per-seed layout "mean (s42 / s1234 / s2024)".

Run: .venv/bin/python -m pytest tests/test_make_summary.py -v   (pandas + pyyaml)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import make_summary as ms  # noqa: E402


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data) if path.suffix == ".yaml" else json.dumps(data))


def _cfg(seed, objective="sft", teacher="none", dataset="tooluse", stage=1, init=None):
    return {"method": {"objective": objective, "tuning": "lora", "teacher": teacher},
            "model": {"id": "Qwen/Qwen2.5-7B-Instruct", "scale": "7b"},
            "data": {"dataset": dataset, "stage": stage, "init_adapter": init},
            "train": {"seed": seed}, "eval": {"scorer": "strict", "forgetting": {"num_fewshot": None}}}


def _res(scores):
    return {"accuracy": sum(scores) / len(scores), "n_effective": len(scores), "wilson95": [0, 1],
            "per_sample_scores": scores}


def test_cell_layout():
    assert ms.cell({1234: 0.2, 42: 0.1, 2024: 0.3}) == "0.200 (0.100 / 0.200 / 0.300)"
    assert ms.cell({42: 0.25}) == "0.250"
    assert ms.cell({}) == ""


def test_summary_from_collected_tables(tmp_path):
    root = tmp_path / "runs"
    for prefix, obj, teacher, s1, s2 in (("sft_lora", "sft", "none", [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1]),
                                         ("sdft_ema_lora", "sdft", "ema", [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0])):
        for seed in (42, 1234, 2024):
            r1, r2 = f"{prefix}_7b_tooluse_s1_seed{seed}", f"{prefix}_7b_science_s2_seed{seed}"
            _write(root / r1 / "resolved_config.yaml", _cfg(seed, obj, teacher))
            _write(root / r1 / "eval/final/tooluse_holdout/eval_results.json", _res(s1))
            _write(root / r2 / "resolved_config.yaml",
                   _cfg(seed, obj, teacher, "science", 2, f"runs/{r1}/lora_adapter"))
            _write(root / r2 / "eval/final/tooluse_holdout/eval_results.json", _res(s2))
            _write(root / r2 / "eval/final/science_eval_data/eval_results.json", _res([1, 0, 1, 0, 1, 0]))
    out = tmp_path / "analysis"
    for cmd in (["collect_results.py", "--root", str(root), "--out", str(out)],
                ["make_summary.py", "--analysis", str(out), "--wandb", str(tmp_path / "none.json")]):
        r = subprocess.run([sys.executable, str(REPO / cmd[0]), *cmd[1:]], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stdout + r.stderr

    md = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| SFT | 1 | 0.500 (0.500 / 0.500 / 0.500) |" in md        # 3/6 right after stage 1
    # SFT holdout: 0.500 -> 0.500, BWT 0, forgot 1 of 3 right per seed, gained 1 of 3 wrong per seed
    assert "| tooluse/holdout | SFT | 0.500 (0.500 / 0.500 / 0.500) | 0.500 (0.500 / 0.500 / 0.500) | +0.0" in md
    assert "3/9 (33.3%) | 3/9 (33.3%)" in md
    # exact 2/6 = 0.333, not the 4-dp 0.3333 averaged
    assert "| SDFT-EMA | 1 | 0.333 (0.333 / 0.333 / 0.333) |" in md
    assert "(no adapter_distance.csv)" in md and "(no wandb_export/runs_index.json)" in md
