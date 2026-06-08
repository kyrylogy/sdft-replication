# Extract a quantitative + qualitative report from one eval run directory,
# or a movement-table comparison from two runs (typically baseline vs ceiling
# on the SAME dataset). Reads eval_results.json + eval_responses.json that
# eval_tooluse.py writes; produces report.json (structured) and report.md
# (tables + concrete examples) alongside.
#
# Usage:
#   python report_extract.py baselines/<run>                     # single-run report
#   python report_extract.py --compare baselines/<base> baselines/<ceil>   # base->ceil movement

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datasets import load_from_disk


def extract_actions(text):
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text):
    blocks = re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL)
    d = {}
    for b in blocks:
        try:
            d.update(json.loads(b))
        except json.JSONDecodeError:
            continue
    return d


def gold_inputs(golden_answer):
    d = {}
    for step in golden_answer:
        try:
            d.update(json.loads(step["Action_Input"]))
        except json.JSONDecodeError:
            continue
    return d


def classify(row):
    """One of: correct, no_action, right_action_wrong_inputs, dup_count,
    partial_overlap, action_wrong."""
    if row["correct"]:
        return "correct"
    pa = extract_actions(row["response"])
    pi = extract_action_inputs(row["response"])
    ga = [g["Action"] for g in row["golden_answer"]]
    gi = gold_inputs(row["golden_answer"])
    if not pa:
        return "no_action"
    if Counter(pa) == Counter(ga) and pi != gi:
        return "right_action_wrong_inputs"
    if Counter(pa) != Counter(ga):
        gs, ps = set(ga), set(pa)
        if gs == ps:
            return "dup_count"
        if gs & ps:
            return "partial_overlap"
        return "action_wrong"
    return "other"


def input_mismatch_breakdown(row):
    """Within right_action_wrong_inputs, classify HOW the inputs differ."""
    pi = extract_action_inputs(row["response"])
    gi = gold_inputs(row["golden_answer"])
    missing = set(gi) - set(pi)
    extra = set(pi) - set(gi)
    common = set(gi) & set(pi)
    value_diff = {k for k in common if pi[k] != gi[k]}
    if missing and not extra and not value_diff:
        return "missing_default_keys"
    if extra and not missing and not value_diff:
        return "extra_keys"
    if value_diff and not missing and not extra:
        return "value_mismatch"
    return "mixed"


def load_run(run_dir):
    results = json.load(open(os.path.join(run_dir, "eval_results.json")))
    responses = json.load(open(os.path.join(run_dir, "eval_responses.json")))
    eval_data_path = results["config"].get("eval_data") or "data/tooluse_data/eval_data"
    ds = load_from_disk(eval_data_path)
    api_names = ds["name"]
    return results, responses, api_names


def single_run_report(run_dir):
    results, responses, api_names = load_run(run_dir)
    n = len(responses)
    correct = sum(r["correct"] for r in responses)

    # Failure-mode buckets
    buckets = Counter()
    input_mismatch = Counter()
    for r in responses:
        bucket = classify(r)
        buckets[bucket] += 1
        if bucket == "right_action_wrong_inputs":
            input_mismatch[input_mismatch_breakdown(r)] += 1

    # Per-API
    per_api = defaultdict(lambda: {"n": 0, "c": 0})
    for r, name in zip(responses, api_names):
        per_api[name]["n"] += 1
        per_api[name]["c"] += r["correct"]
    per_api_sorted = sorted(
        per_api.items(), key=lambda x: -x[1]["c"] / x[1]["n"]
    )

    # Step-count breakdown
    steps = Counter(len(r["golden_answer"]) for r in responses)
    single_n = steps[1]
    multi_n = n - single_n
    single_c = sum(r["correct"] for r in responses if len(r["golden_answer"]) == 1)
    multi_c = correct - single_c

    # Examples: 2 failures per bucket, 2 successes
    examples = {b: [] for b in buckets}
    examples["correct"] = []
    for i, r in enumerate(responses):
        bucket = classify(r)
        if len(examples.get(bucket, [])) >= 2:
            continue
        examples.setdefault(bucket, []).append({
            "row_index": i,
            "api": api_names[i],
            "instruction": (r.get("golden_answer") and ""),  # filled below
            "golden_actions": [g["Action"] for g in r["golden_answer"]],
            "golden_inputs": gold_inputs(r["golden_answer"]),
            "predicted_actions": extract_actions(r["response"]),
            "predicted_inputs": extract_action_inputs(r["response"]),
            "response_first_400_chars": r["response"][:400],
        })
    # Fetch instruction text from the dataset
    eval_data_path = results["config"].get("eval_data") or "data/tooluse_data/eval_data"
    ds = load_from_disk(eval_data_path)
    instructions = ds["instruction"]
    for bucket_examples in examples.values():
        for ex in bucket_examples:
            ex["instruction"] = instructions[ex["row_index"]]

    report = {
        "source_run": run_dir,
        "config": results["config"],
        "headline": {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
        },
        "failure_modes": dict(buckets),
        "input_mismatch_breakdown": dict(input_mismatch),
        "step_count": {
            "single_step": {"n": single_n, "correct": single_c,
                            "accuracy": single_c / max(single_n, 1)},
            "multi_step": {"n": multi_n, "correct": multi_c,
                           "accuracy": multi_c / max(multi_n, 1)},
            "step_distribution": dict(sorted(steps.items())),
        },
        "per_api": [
            {"api": name, "n": v["n"], "correct": v["c"],
             "accuracy": v["c"] / v["n"]}
            for name, v in per_api_sorted
        ],
        "examples": examples,
    }
    return report


def comparison_report(base_dir, ceil_dir):
    base_results, base_resp, base_apis = load_run(base_dir)
    ceil_results, ceil_resp, ceil_apis = load_run(ceil_dir)
    assert len(base_resp) == len(ceil_resp), (
        f"row counts differ: {len(base_resp)} vs {len(ceil_resp)} — "
        f"comparison requires same eval_data"
    )

    movement = Counter()
    flips_wrong_to_right = []
    flips_right_to_wrong = []
    for i, (b, c) in enumerate(zip(base_resp, ceil_resp)):
        movement[(b["correct"], c["correct"])] += 1
        if not b["correct"] and c["correct"]:
            flips_wrong_to_right.append(i)
        elif b["correct"] and not c["correct"]:
            flips_right_to_wrong.append(i)

    return {
        "base_run": base_dir,
        "ceiling_run": ceil_dir,
        "n": len(base_resp),
        "base_accuracy": base_results["accuracy"],
        "ceiling_accuracy": ceil_results["accuracy"],
        "absolute_gain": ceil_results["accuracy"] - base_results["accuracy"],
        "movement": {
            "wrong_to_right_gained": movement[(False, True)],
            "right_to_right_kept": movement[(True, True)],
            "right_to_wrong_lost": movement[(True, False)],
            "wrong_to_wrong_still": movement[(False, False)],
        },
        "flips_wrong_to_right_indices": flips_wrong_to_right[:20],  # cap for readability
        "flips_right_to_wrong_indices": flips_right_to_wrong,
    }


def render_markdown(report):
    """Render a single-run report.json as a markdown report."""
    lines = []
    h = report["headline"]
    cfg = report["config"]
    lines.append(f"# Eval report — `{report['source_run']}`\n")
    lines.append(f"**Model**: `{cfg.get('model_path')}`")
    if cfg.get("adapter_path"):
        lines.append(f"**Adapter**: `{cfg['adapter_path']}`")
    lines.append(f"**Eval data**: `{cfg.get('eval_data')}`")
    lines.append(f"**Teacher-ceiling mode**: `{cfg.get('teacher_ceiling')}`"
                 f"{' (demo_source=' + str(cfg.get('demo_source')) + ')' if cfg.get('teacher_ceiling') else ''}")
    lines.append(f"**Engine / device / dtype**: `{cfg.get('engine')}` / `{cfg.get('device')}` / `{cfg.get('dtype')}`\n")

    lines.append(f"## Headline\n")
    lines.append(f"- n = **{h['n']}**")
    lines.append(f"- correct = **{h['correct']}**")
    lines.append(f"- accuracy = **{h['accuracy']*100:.2f}%**\n")

    lines.append(f"## Failure-mode buckets\n")
    lines.append("| bucket | count | share |")
    lines.append("|---|---:|---:|")
    for bucket, c in sorted(report["failure_modes"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{bucket}` | {c} | {c/h['n']*100:.1f}% |")
    if report["input_mismatch_breakdown"]:
        lines.append("\n**Within `right_action_wrong_inputs` (input dict mismatch types):**\n")
        lines.append("| kind | count |")
        lines.append("|---|---:|")
        for k, c in sorted(report["input_mismatch_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {c} |")

    sc = report["step_count"]
    lines.append(f"\n## Step-count breakdown\n")
    lines.append("| group | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| single-step | {sc['single_step']['n']} | {sc['single_step']['correct']} | {sc['single_step']['accuracy']*100:.1f}% |")
    lines.append(f"| multi-step  | {sc['multi_step']['n']} | {sc['multi_step']['correct']} | {sc['multi_step']['accuracy']*100:.1f}% |")
    lines.append(f"\nGolden step-count distribution: `{sc['step_distribution']}`\n")

    lines.append(f"## Per-API\n")
    lines.append("| API | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for row in report["per_api"]:
        lines.append(f"| {row['api']} | {row['n']} | {row['correct']} | {row['accuracy']*100:.1f}% |")

    lines.append(f"\n## Concrete examples (up to 2 per bucket)\n")
    for bucket, exs in report["examples"].items():
        if not exs:
            continue
        lines.append(f"### `{bucket}`\n")
        for ex in exs:
            lines.append(f"**row {ex['row_index']} | api={ex['api']}**\n")
            lines.append(f"- INSTRUCTION: {ex['instruction']}")
            lines.append(f"- GOLD actions: `{ex['golden_actions']}`")
            lines.append(f"- GOLD inputs:  `{ex['golden_inputs']}`")
            lines.append(f"- PRED actions: `{ex['predicted_actions']}`")
            lines.append(f"- PRED inputs:  `{ex['predicted_inputs']}`")
            lines.append(f"- RESPONSE (first 400c):\n\n```\n{ex['response_first_400_chars']}\n```\n")
    return "\n".join(lines)


def render_comparison_markdown(cmp):
    h = cmp
    lines = [f"# Movement: `{cmp['base_run']}` → `{cmp['ceiling_run']}`\n"]
    lines.append(f"- n = **{h['n']}**")
    lines.append(f"- base accuracy:    **{h['base_accuracy']*100:.2f}%**")
    lines.append(f"- ceiling accuracy: **{h['ceiling_accuracy']*100:.2f}%**")
    lines.append(f"- Δ = **{h['absolute_gain']*100:+.2f}** points\n")
    m = cmp["movement"]
    lines.append("## Movement table\n")
    lines.append("| transition | count |")
    lines.append("|---|---:|")
    lines.append(f"| wrong → right (gained) | {m['wrong_to_right_gained']} |")
    lines.append(f"| right → right (kept)   | {m['right_to_right_kept']} |")
    lines.append(f"| right → wrong (lost)   | {m['right_to_wrong_lost']} |")
    lines.append(f"| wrong → wrong (still)  | {m['wrong_to_wrong_still']} |")
    if cmp["flips_right_to_wrong_indices"]:
        lines.append(f"\n**Right→Wrong row indices** (regressions, worth inspecting):"
                     f" `{cmp['flips_right_to_wrong_indices']}`")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", help="Single run directory (eval_tooluse.py output)")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "CEIL"),
                   help="Compute base→ceiling movement table over the same eval set")
    args = p.parse_args()

    if args.compare:
        base_dir, ceil_dir = args.compare
        cmp = comparison_report(base_dir, ceil_dir)
        out_json = os.path.join(ceil_dir, "comparison_vs_base.json")
        out_md = os.path.join(ceil_dir, "comparison_vs_base.md")
        with open(out_json, "w") as f:
            json.dump(cmp, f, indent=2)
        with open(out_md, "w") as f:
            f.write(render_comparison_markdown(cmp))
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")
        return

    if not args.run:
        p.error("Provide a run dir or --compare BASE CEIL")
    report = single_run_report(args.run)
    out_json = os.path.join(args.run, "report.json")
    out_md = os.path.join(args.run, "report.md")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_md, "w") as f:
        f.write(render_markdown(report))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
