#!/usr/bin/env python
"""Export the small, publishable per-run records of runs/ into runs_export/.

runs/ also holds adapters and checkpoints (~10 GB) and is not committed. This copies only the
records the analysis reads, in runs/'s layout, so the tables rebuild from git alone:

    python collect_results.py --root runs_export --out /tmp/check   # == analysis/*.csv
    python stats_final.py     --root runs_export --out /tmp/check   # == analysis/final_stats.txt

Per run:
  resolved_config.yaml, eval/**/resolved_config.yaml           run stamps (config, git sha, libs)
  eval/<checkpoint>/<set>/eval_results.json                     accuracy, Wilson CI, per_sample_scores
  eval/forgetting*/**/results*.json                             lm-eval battery outputs
  eval/<final|base_anchor|ceiling>/<set>/eval_responses.json    generations (--responses final)

What changes on the way out (everything else is copied byte for byte):
  - the training host's absolute repo path (stamped as _derived.repo) becomes repo-relative, in
    file contents and in lm-eval's model-path directory names
    (__home__<user>__...__sdft-replication__runs__<run>__lora_adapter -> runs__<run>__lora_adapter);
  - the stamped host name (_provenance.host) and lm-eval's pretty_env_info become null.
Nothing is written if a stamped path, user or host name would survive in the records, or if no
run is stamped. runs_index.has_adapter is False when rebuilt from the export (no weights copied).

Usage:
    python export_runs.py                     # runs/ -> runs_export/  (~26 MB, with generations)
    python export_runs.py --responses none    # records only (~2 MB)
    python export_runs.py --responses all     # generations of every checkpoint (~100 MB)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
FINAL = {"final", "base_anchor", "ceiling"}   # eval dirs whose generations --responses final keeps
HOME_PATH = re.compile(r"(?<![\w.~-])(?:/home/|/Users/|[A-Za-z]:(?:\\\\|\\|/)Users)")
ENV_INFO = re.compile(r'"pretty_env_info":\s*"(?:[^"\\]|\\.)*"')


def select(run_dir, responses):
    """[(kind, path)] for one run: the same globs collect_results.load_runs reads."""
    ev = run_dir / "eval"
    files = [("configs", run_dir / "resolved_config.yaml")]
    files += [("configs", p) for p in sorted(ev.glob("**/resolved_config.yaml"))]
    files += [("eval_results", p) for p in sorted(ev.glob("*/*/eval_results.json"))]
    files += [("lmeval", p) for p in sorted(ev.glob("forgetting*/**/results*.json"))]
    if responses != "none":
        files += [("responses", p) for p in sorted(ev.glob("*/*/eval_responses.json"))
                  if responses == "all" or p.parent.parent.name in FINAL]
    return files


def stamps(run_dirs):
    """(repo paths, host names) stamped in the runs' resolved_config.yaml files."""
    repos, hosts = set(), set()
    for d in run_dirs:
        for cfg_path in [d / "resolved_config.yaml", *(d / "eval").glob("**/resolved_config.yaml")]:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            repo = (cfg.get("_derived") or {}).get("repo")
            host = (cfg.get("_provenance") or {}).get("host")
            if repo:
                repos.add(str(repo).rstrip("/"))
            if host:
                hosts.add(str(host))
    return sorted(repos, key=len, reverse=True), sorted(hosts)


def make_scrub(repo_paths, hosts):
    """str -> str for one file's text: repo paths -> relative (only where a path ends), lm-eval's
    '__'-joined form of them -> '', `host: <stamped host>` -> null, pretty_env_info -> null."""
    rules = []
    for p in repo_paths:
        rules.append((re.compile(r"(?<![\w.~/-])" + re.escape(p) + r"(/|(?=[\s\"',;)\]}]|$))", re.M),
                      lambda m: "" if m.group(1) else "."))
        rules.append((re.compile(r"(?<![\w.~-])" + re.escape("__" + p.strip("/").replace("/", "__") + "__")),
                      lambda m: ""))
    if hosts:
        rules.append((re.compile(r"(?m)^([ \t]*host:[ \t]*)(?:" + "|".join(map(re.escape, hosts)) + r")(?=[ \t]*\r?$)"),
                      lambda m: m.group(1) + "null"))
    rules.append((ENV_INFO, lambda m: '"pretty_env_info": null'))

    def scrub(text):
        for rx, repl in rules:
            text = rx.sub(repl, text)
        return text
    return scrub


def leak_tokens(repo_paths):
    """Path strings that must not survive anywhere: the repo paths and their user's home prefix."""
    tokens = set(repo_paths)
    for p in repo_paths:
        parts = p.strip("/").split("/")
        if len(parts) > 1 and parts[0] in ("home", "Users"):
            tokens |= {f"/{parts[0]}/{parts[1]}/", f"__{parts[0]}__{parts[1]}__"}
    return tokens


def problem(src, kind, text, dst, tokens, hosts):
    """'' if the scrubbed file still parses, keeps its per_sample_scores and leaks nothing.
    Generations are only checked for stamped paths: a host name or '/home/' can occur in a prompt."""
    try:
        if src.suffix == ".json":
            new = json.loads(text)
            old = json.loads(src.read_bytes().decode("utf-8"))
            if isinstance(old, dict) and old.get("per_sample_scores") != new.get("per_sample_scores"):
                return "per_sample_scores changed"
        else:
            yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as e:
        return f"does not parse after scrubbing ({e.__class__.__name__})"
    leak = next((t for t in tokens if t in text or t in dst.as_posix()), None)
    if leak is None and kind != "responses":
        leak = next((h for h in hosts if h in text), None)
        if leak is None and HOME_PATH.search(text):
            leak = HOME_PATH.search(text).group(0)
    return f"still contains {leak!r}" if leak else ""


def main():
    ap = argparse.ArgumentParser(description="Copy the publishable per-run records of runs/ "
                                             "(stamps, eval results, lm-eval outputs, generations)")
    ap.add_argument("--root", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "runs_export"))
    ap.add_argument("--responses", choices=["none", "final", "all"], default="final",
                    help="which eval_responses.json (generations) to include")
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents or out in root.parents:
        sys.exit(f"[export] --out {out} must not overlap --root {root}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(f"[export] {out} exists and is not an empty directory; remove it first (rm -rf {args.out})")
    run_dirs = [p.parent for p in sorted(root.glob("*/resolved_config.yaml"))]
    if not run_dirs:
        sys.exit(f"[export] no runs with resolved_config.yaml under {root}")

    repo_paths, hosts = stamps(run_dirs)
    if not repo_paths:
        sys.exit("[export] no run stamps _derived.repo, so the absolute paths to rewrite are unknown")
    scrub, tokens = make_scrub(repo_paths, hosts), leak_tokens(repo_paths)
    plan, problems, renamed = [], [], set()
    for d in run_dirs:
        for kind, src in select(d, args.responses):
            rel = src.relative_to(root)
            dst = Path(*(scrub(part) for part in rel.parts))
            try:
                raw = src.read_bytes().decode("utf-8")
            except UnicodeDecodeError as e:
                problems.append(f"{rel.as_posix()}: not UTF-8 (byte {e.start})")
                continue
            text = scrub(raw)   # bytes in/out: line endings and untouched text stay identical
            err = problem(src, kind, text, dst, tokens, hosts)
            if err:
                problems.append(f"{rel.as_posix()}: {err}")
            plan.append((kind, dst, text, text != raw))
            renamed |= {(a, b) for a, b in zip(rel.parts, dst.parts) if a != b}
    if problems:
        print(f"[export] {len(problems)} problem(s), nothing written:", *problems, sep="\n  ")
        sys.exit(1)

    for _, dst, text, _ in plan:
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        (out / dst).write_bytes(text.encode("utf-8"))

    files, size = Counter(), Counter()
    for kind, _, text, _ in plan:
        files[kind] += 1
        size[kind] += len(text.encode("utf-8"))
    print(f"[export] {len(run_dirs)} runs: {root} -> {out}  (--responses {args.responses})")
    for kind in ("configs", "eval_results", "lmeval", "responses"):
        print(f"  {kind:13} {files[kind]:>5} files {size[kind]:>13,} bytes")
    print(f"[scrub] repo path(s) {repo_paths} -> relative, host(s) {hosts} -> null: "
          f"{sum(changed for *_, changed in plan)} files rewritten, {len(renamed)} directories renamed")


if __name__ == "__main__":
    main()
