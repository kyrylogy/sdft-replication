"""Tests for export_runs.py: what leaves runs/, and that the tables still rebuild from it.

Builds a synthetic runs/ tree whose stamps carry a fake training-host path
(/home/alice@lab.example/projects/sdft-replication) and host name (gpu-box-01), exports it, and
checks that path, user and host are gone from contents and paths, per_sample_scores are
unchanged, weights never leave, --responses picks the right generations, unsafe trees are
refused, and collect_results.py builds the same tables from runs/ and from the export.

Run: .venv/bin/python -m pytest tests/test_export_runs.py -v   (pyyaml + stdlib only)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
HOST = "/home/alice@lab.example/projects/sdft-replication"
S1, S2 = "sft_lora_7b_tooluse_s1_seed42", "sft_lora_7b_science_s2_seed42"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _cfg(run, *, stage=1, dataset="tooluse", init=None, repo=HOST):
    return yaml.safe_dump({
        "method": {"objective": "sft", "tuning": "lora", "teacher": "none"},
        "model": {"id": "Qwen/Qwen2.5-7B-Instruct", "scale": "7b"},
        "data": {"dataset": dataset, "stage": stage, "init_adapter": init},
        "train": {"seed": 42},
        "eval": {"scorer": "strict", "forgetting": {"num_fewshot": None}},
        "_derived": {"output_dir": f"{repo}/runs/{run}", "repo": repo},
        "_provenance": {"host": "gpu-box-01", "git_sha": "0123abcd"},
    }, sort_keys=False)


def _acc(run, scores):
    return json.dumps({"accuracy": sum(scores) / len(scores), "n_effective": len(scores),
                       "wilson95": [0.1, 0.9], "per_sample_scores": scores,
                       "adapter": f"{HOST}/runs/{run}/lora_adapter"}, indent=2)


@pytest.fixture
def runs(tmp_path):
    """Stage 1 (final + step50 evals, generations, an lm-eval battery, weights) and stage 2."""
    root = tmp_path / "runs"
    ev = root / S1 / "eval"
    _write(root / S1 / "resolved_config.yaml", _cfg(S1))
    _write(ev / "final/tooluse_holdout/eval_results.json", _acc(S1, [1, 0, 1, 0]))
    _write(ev / "final/tooluse_holdout/eval_responses.json", json.dumps([{"response": "final"}]))
    _write(ev / "step50/tooluse_holdout/eval_results.json", _acc(S1, [1, 0, 0, 0]))
    _write(ev / "step50/tooluse_holdout/eval_responses.json", json.dumps([{"response": "step50"}]))
    lm = f"__home__alice@lab.example__projects__sdft-replication__runs__{S1}__lora_adapter"
    _write(ev / "forgetting" / lm / "results_2026-01-01T00-00-00.json", json.dumps({
        "results": {"hellaswag": {"acc_norm,none": 0.8, "acc_norm_stderr,none": 0.01}},
        "config": {"model_args": f"pretrained=Qwen/Qwen2.5-7B-Instruct,peft={HOST}/runs/{S1}/lora_adapter"},
        "model_name_sanitized": lm,
        "pretty_env_info": "OS: Ubuntu 24.04\nGPU 0: NVIDIA H100 \"NVL\""}, indent=2))
    _write(root / S1 / "lora_adapter/adapter_model.safetensors", "weights")
    _write(root / S1 / "checkpoint-50/optimizer.pt", "optimizer state")
    _write(root / S2 / "resolved_config.yaml", _cfg(S2, stage=2, dataset="science", init=f"runs/{S1}/lora_adapter"))
    _write(root / S2 / "eval/final/tooluse_holdout/eval_results.json", _acc(S2, [1, 0, 0, 0]))
    return root


def _export(root, out, *args):
    return subprocess.run([sys.executable, str(REPO / "export_runs.py"), "--root", str(root),
                           "--out", str(out), *args], capture_output=True, text=True, cwd=REPO)


def test_host_paths_scrubbed_scores_kept_weights_left_behind(runs, tmp_path):
    out = tmp_path / "runs_export"
    r = _export(runs, out)
    assert r.returncode == 0, r.stdout + r.stderr
    files = [p for p in out.rglob("*") if p.is_file()]
    for p in files:
        rel = p.relative_to(out).as_posix()
        assert "alice" not in rel and "home" not in rel
        text = p.read_text()
        assert "alice" not in text and "/home/" not in text and "gpu-box-01" not in text, rel
    assert not any(p.suffix in (".safetensors", ".pt") for p in files)

    ev = out / S1 / "eval"
    rec = json.loads((ev / "final/tooluse_holdout/eval_results.json").read_text())
    assert rec["adapter"] == f"runs/{S1}/lora_adapter"
    assert rec["per_sample_scores"] == [1, 0, 1, 0]
    cfg = yaml.safe_load((out / S1 / "resolved_config.yaml").read_text())
    assert cfg["_derived"] == {"output_dir": f"runs/{S1}", "repo": "."}
    assert cfg["_provenance"] == {"host": None, "git_sha": "0123abcd"}
    (lm_dir,) = (ev / "forgetting").iterdir()
    assert lm_dir.name == f"runs__{S1}__lora_adapter"
    battery = json.loads(next(lm_dir.glob("results*.json")).read_text())
    assert battery["config"]["model_args"].endswith(f"peft=runs/{S1}/lora_adapter")
    assert battery["pretty_env_info"] is None and battery["results"]["hellaswag"]["acc_norm,none"] == 0.8


def test_responses_modes(runs, tmp_path):
    def exported(mode):
        out = tmp_path / f"export_{mode}"
        assert _export(runs, out, "--responses", mode).returncode == 0
        return sorted(p.parent.parent.name for p in out.rglob("eval_responses.json"))
    assert exported("none") == []
    assert exported("final") == ["final"]
    assert exported("all") == ["final", "step50"]


def test_refuses_a_non_empty_out_dir(runs, tmp_path):
    out = tmp_path / "runs_export"
    _write(out / "keep.txt", "x")
    r = _export(runs, out)
    assert r.returncode != 0 and "not an empty directory" in r.stdout + r.stderr
    assert (out / "keep.txt").exists()


def test_refuses_unstamped_runs_and_leftover_home_paths(runs, tmp_path):
    # no _derived.repo anywhere: the paths to rewrite are unknown, so nothing is exported
    bare = tmp_path / "bare"
    _write(bare / S1 / "resolved_config.yaml", yaml.safe_dump({"method": {"objective": "sft"}}))
    r = _export(bare, tmp_path / "out_bare")
    assert r.returncode != 0 and not (tmp_path / "out_bare").exists()

    # a record pointing at another checkout survives the rewrite: refused, nothing written
    _write(runs / S1 / "eval/step50/tooluse_holdout/eval_results.json",
           json.dumps({"accuracy": 0.25, "per_sample_scores": [1, 0, 0, 0],
                       "adapter": "/home/bob/other-checkout/runs/x/lora_adapter"}))
    r = _export(runs, tmp_path / "out_leak")
    assert r.returncode != 0 and "/home/" in r.stdout and not (tmp_path / "out_leak").exists()


def test_collect_results_builds_the_same_tables_from_the_export(runs, tmp_path):
    out = tmp_path / "runs_export"
    assert _export(runs, out).returncode == 0
    tables = {}
    for name, root in (("runs", runs), ("export", out)):
        dst = tmp_path / f"analysis_{name}"
        r = subprocess.run([sys.executable, str(REPO / "collect_results.py"), "--root", str(root),
                            "--out", str(dst)], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stdout + r.stderr
        tables[name] = {n: sorted((dst / n).read_text().splitlines())   # row order follows the filesystem
                        for n in ("retention.csv", "results_long.csv", "forgetting.csv")}
    assert tables["runs"] == tables["export"]
    assert any("tooluse/holdout" in line for line in tables["export"]["retention.csv"])
