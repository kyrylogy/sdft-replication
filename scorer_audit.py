"""Audit current vs strict scorer on every eval_responses.json under baselines/."""

import glob
import json
import os
import re
from collections import Counter


# ---------------------------------------------------------------------------
# CURRENT scorer (copied verbatim from eval_tooluse.py)
# ---------------------------------------------------------------------------

def extract_actions(text):
    """Extract all actions from model response."""
    return re.findall(r'Action:\s*(\w+)', text)


def extract_action_inputs(text):
    """Extract and merge all action inputs from model response."""
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            continue
    return combined_dict


def score_current(response, golden_answer):
    """Return (verdict_str, pred_actions, pred_inputs_merged, gt_actions, gt_inputs_merged)."""
    pred_actions = extract_actions(response)
    pred_inputs = extract_action_inputs(response)

    gt_actions = [item['Action'] for item in golden_answer]
    gt_inputs = {}
    for item in golden_answer:
        try:
            gt_inputs.update(json.loads(item['Action_Input']))
        except Exception:
            pass

    actions_match = Counter(pred_actions) == Counter(gt_actions)
    inputs_match = pred_inputs == gt_inputs
    verdict = "correct" if (actions_match and inputs_match) else "incorrect"
    return verdict, pred_actions, pred_inputs, gt_actions, gt_inputs


# ---------------------------------------------------------------------------
# STRICT scorer
# ---------------------------------------------------------------------------

def extract_action_pairs(text):
    """Extract ordered (action_name, parsed_input_dict) pairs.

    Skips pairs where the JSON parse fails. Returns list of tuples
    [(action_name, dict), ...] in order of appearance.
    """
    pairs = []
    pattern = re.compile(r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{.*?\})', re.DOTALL)
    for m in pattern.finditer(text):
        name = m.group(1)
        raw = m.group(2)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        pairs.append((name, parsed))
    return pairs


def score_strict(response, golden_answer):
    """Strict element-wise pairing scorer.

    Builds gold pairs as [(item['Action'], json.loads(item['Action_Input']))
    for item in golden_answer]. If ANY gold Action_Input fails to parse,
    returns ("ungradeable", ...) so the caller can drop the row.

    Otherwise compares pred pairs vs gold pairs element-wise:
      - same length
      - each (action_name, input_dict) matches by exact name + dict equality
    No Counter, no merging.
    """
    # Build gold pairs strictly.
    gold_pairs = []
    for item in golden_answer:
        try:
            parsed = json.loads(item['Action_Input'])
        except Exception:
            return "ungradeable", [], []
        gold_pairs.append((item['Action'], parsed))

    pred_pairs = extract_action_pairs(response)

    if len(pred_pairs) != len(gold_pairs):
        return "incorrect", pred_pairs, gold_pairs

    for (pn, pi), (gn, gi) in zip(pred_pairs, gold_pairs):
        if pn != gn or pi != gi:
            return "incorrect", pred_pairs, gold_pairs

    return "correct", pred_pairs, gold_pairs


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------

def truncate(s, n=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= n else s[:n] + "...[truncated]"


def audit_file(path):
    with open(path) as f:
        data = json.load(f)

    n_rows = len(data)
    gold_parse_failed = 0
    current_correct = 0
    strict_correct = 0
    both_correct = 0
    both_incorrect = 0
    fp_current_yes_strict_no = 0
    fn_current_no_strict_yes = 0
    disagreement_rows = []  # list of dicts for first interesting disagreements

    per_row = []

    for idx, row in enumerate(data):
        response = row.get("response", "")
        golden_answer = row.get("golden_answer", [])

        # Pre-check: any gold Action_Input that fails to parse means we cannot
        # compare strictly; we drop the row from agreement stats.
        gold_ok = True
        for item in golden_answer:
            try:
                json.loads(item['Action_Input'])
            except Exception:
                gold_ok = False
                break
        if not gold_ok:
            gold_parse_failed += 1
            per_row.append({"row_index": idx, "current": None, "strict": None,
                            "note": "gold_parse_failed"})
            continue

        cur_verdict, pred_actions, pred_inputs_merged, gt_actions, gt_inputs_merged = score_current(
            response, golden_answer
        )
        strict_verdict, pred_pairs, gold_pairs = score_strict(response, golden_answer)

        if cur_verdict == "correct":
            current_correct += 1
        if strict_verdict == "correct":
            strict_correct += 1
        if cur_verdict == "correct" and strict_verdict == "correct":
            both_correct += 1
        if cur_verdict == "incorrect" and strict_verdict == "incorrect":
            both_incorrect += 1
        if cur_verdict == "correct" and strict_verdict == "incorrect":
            fp_current_yes_strict_no += 1
        if cur_verdict == "incorrect" and strict_verdict == "correct":
            fn_current_no_strict_yes += 1

        if cur_verdict != strict_verdict:
            disagreement_rows.append({
                "row_index": idx,
                "current_verdict": cur_verdict,
                "strict_verdict": strict_verdict,
                "pred_actions": pred_actions,
                "pred_inputs_per_step": [p[1] for p in pred_pairs],
                "pred_inputs_merged": pred_inputs_merged,
                "gold_actions": gt_actions,
                "gold_inputs_per_step": [item['Action_Input'] for item in golden_answer],
                "gold_inputs_merged": gt_inputs_merged,
                "response_excerpt": truncate(response, 800),
            })

        per_row.append({"row_index": idx, "current": cur_verdict, "strict": strict_verdict})

    n_eff = n_rows - gold_parse_failed
    n_disagreements = fp_current_yes_strict_no + fn_current_no_strict_yes

    return {
        "response_file": path,
        "n_rows": n_rows,
        "gold_parse_failed": gold_parse_failed,
        "n_effective_rows": n_eff,
        "current_correct": current_correct,
        "strict_correct": strict_correct,
        "current_accuracy": current_correct / n_eff if n_eff else None,
        "strict_accuracy": strict_correct / n_eff if n_eff else None,
        "n_disagreements": n_disagreements,
        "false_positives_current_says_yes_strict_says_no": fp_current_yes_strict_no,
        "false_negatives_current_says_no_strict_says_yes": fn_current_no_strict_yes,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "disagreement_rows_top5": disagreement_rows[:5],
        "disagreement_row_indices": [d["row_index"] for d in disagreement_rows],
        "per_row": per_row,
    }


def main():
    root = "/Users/kyrylogy/Projects/University/WS25/Thesis/Self-Distillation"
    pattern = os.path.join(root, "baselines", "*", "eval_responses.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} eval_responses.json files:")
    for p in files:
        print(f"  {p}")
    print()

    all_results = []
    for path in files:
        res = audit_file(path)
        all_results.append(res)
        print("=" * 80)
        print(f"FILE: {res['response_file']}")
        print(f"  rows               = {res['n_rows']}")
        print(f"  gold_parse_failed  = {res['gold_parse_failed']}")
        print(f"  effective rows     = {res['n_effective_rows']}")
        print(f"  current correct    = {res['current_correct']}  "
              f"(acc={res['current_accuracy']})")
        print(f"  strict  correct    = {res['strict_correct']}  "
              f"(acc={res['strict_accuracy']})")
        print(f"  disagreements      = {res['n_disagreements']}")
        print(f"    FP (current=yes, strict=no) = {res['false_positives_current_says_yes_strict_says_no']}")
        print(f"    FN (current=no,  strict=yes)= {res['false_negatives_current_says_no_strict_says_yes']}")
        if res["disagreement_rows_top5"]:
            print("  --- top disagreement examples ---")
            for d in res["disagreement_rows_top5"]:
                print(f"    row {d['row_index']}: current={d['current_verdict']} strict={d['strict_verdict']}")
                print(f"      pred_actions = {d['pred_actions']}")
                print(f"      gold_actions = {d['gold_actions']}")
                print(f"      pred_inputs_per_step = {d['pred_inputs_per_step']}")
                print(f"      gold_inputs_per_step = {d['gold_inputs_per_step']}")
                print(f"      pred_inputs_merged   = {d['pred_inputs_merged']}")
                print(f"      gold_inputs_merged   = {d['gold_inputs_merged']}")
        print()

    out_path = os.path.join(root, "scorer_audit_results.json")
    # Strip per_row from JSON dump to keep file readable; keep summary + disagreements.
    dump = []
    for r in all_results:
        rr = dict(r)
        rr.pop("per_row", None)
        dump.append(rr)
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
