#!/usr/bin/env python3
"""Grade the Day 17 lab against rubric.md (100 core pts). Exit 0 if 100/100."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

from pipeline import config
from pipeline.dag import DAG
from pipeline.dataset import build_eval_set, build_preference_pairs, decontaminate
from pipeline.features import naive_leaky_features, point_in_time_features
from pipeline.streaming import MiniTopic, consume_features
from pipeline.traces import load_traces, traces_to_bronze, trace_summary
from pipeline.validate import validate, write_quarantine
import main as pipeline_main


ROOT = Path(__file__).resolve().parent
CORE_TOTAL = 100
scores: list[tuple[int, str, str, int, bool]] = []


def grade(num: int, where: str, criterion: str, pts: int, ok: bool) -> None:
    scores.append((num, where, criterion, pts, ok))


def run_dbt() -> bool:
    dbt_py = ROOT / ".venv-dbt" / "bin" / "dbt"
    if not dbt_py.exists():
        return False
    r = subprocess.run(
        [str(dbt_py), "build"],
        cwd=ROOT / "dbt_project",
        env={"DBT_PROFILES_DIR": ".", **dict(__import__("os").environ)},
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and "PASS=11" in r.stdout


def run_verify() -> tuple[bool, int, int]:
    r = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"DISABLE_PANDERA_IMPORT_WARNING": "True", **dict(__import__("os").environ)},
    )
    out = r.stdout + r.stderr
    passed = "ALL PASS" in out
    if "RESULT:" in out:
        line = [l for l in out.splitlines() if l.startswith("RESULT:")][-1]
        part = line.split("RESULT:")[1].strip().split()[0]
        got, total = part.split("/")
        return passed, int(got), int(total)
    return passed, 0, 16


def check_dag_topo() -> bool:
    order: list[str] = []

    dag = DAG()

    @dag.task("extract")
    def _extract():
        order.append("extract")
        return 1

    @dag.task("validate", upstream=["extract"])
    def _validate(_):
        order.append("validate")
        return 2

    @dag.task("transform", upstream=["validate"])
    def _transform(_):
        order.append("transform")
        return 3

    dag.run()
    return order == ["extract", "validate", "transform"]


def run_grading() -> int:
    # --- Criterion 1: Bronze + Silver dedup (5 dropped) ---
    config.WAREHOUSE.unlink(missing_ok=True)
    stats = pipeline_main.main()
    c1 = (
        stats["rows_in"] == 13
        and stats["dropped_dupes"] == 5
        and stats["rows_out"] == 8
    )
    grade(1, "main.py / verify.py", "Bronze loads raw; Silver dedups order_id (5 dropped)", 8, c1)

    # --- Criterion 2: Gate quarantines exactly 3 ---
    df = pd.read_csv(config.RAW_CSV, dtype=str)
    clean, bad = validate(df)
    n_bad = write_quarantine(bad)
    qfile = config.QUARANTINE
    c2 = len(bad) == 3 and n_bad == 3 and qfile.exists() and len(pd.read_csv(qfile)) == 3
    # one bad row never halts run: main.main() already succeeded above
    grade(2, "pipeline/validate.py", "Gate quarantines exactly 3; bad row never halts run", 10, c2)

    # --- Criterion 3: Gold completed by day; no dup order_id ---
    con = duckdb.connect(str(config.WAREHOUSE))
    (dupes,) = con.execute(
        f"SELECT count(*) - count(DISTINCT order_id) FROM {config.SILVER}"
    ).fetchone()
    total_gold = con.execute(f"SELECT sum(n_orders) FROM {config.GOLD}").fetchone()[0]
    completed = con.execute(
        f"SELECT count(*) FROM {config.SILVER} WHERE status='completed'"
    ).fetchone()[0]
    gold_rows = con.execute(f"SELECT count(*) FROM {config.GOLD}").fetchone()[0]
    con.close()
    c3 = dupes == 0 and total_gold == completed and gold_rows == stats["gold_rows"] == 5
    grade(3, "main.py", "Gold aggregates completed by day; no duplicate order_id", 7, c3)

    # --- Criterion 4: DAG topo order ---
    c4 = check_dag_topo()
    grade(4, "pipeline/dag.py", "Pipeline runs as topologically-ordered DAG", 5, c4)

    # --- Criterion 5: Streaming idempotent ---
    topic = MiniTopic()
    topic.produce("u1", {"event_id": "e1", "amount": 10})
    topic.produce("u1", {"event_id": "e1", "amount": 10})
    topic.produce("u2", {"event_id": "e2", "amount": 5})
    feats = consume_features(topic)
    c5 = feats["u1"]["orders"] == 1 and feats["u2"]["orders"] == 1
    grade(5, "pipeline/streaming.py", "Partition-by-key + idempotent consumer", 8, c5)

    # --- Criterion 6: dbt build ---
    c6 = run_dbt()
    grade(6, "dbt_project/", "dbt build passes (staging→gold + tests + unit test)", 7, c6)

    # --- Flywheel criteria 7-12 ---
    fcon = duckdb.connect(":memory:")
    traces = load_traces()
    n_spans = traces_to_bronze(fcon, traces)
    c7 = n_spans == 21 and n_spans >= len(traces)
    grade(7, "pipeline/traces.py", "gen_ai span trees flattened into Bronze (1 row/span)", 12, c7)

    summary = trace_summary(fcon)
    c8 = (
        len(summary) == 8
        and {"total_tokens", "latency_ms", "outcome"}.issubset(summary.columns)
        and set(summary["outcome"]) == {"ok", "error"}
    )
    grade(8, "flywheel.py", "Per-trace summary: cost + latency + outcome", 6, c8)

    eval_set = build_eval_set(fcon)
    c9 = len(eval_set) == 2 and all(e.get("reference") for e in eval_set)
    grade(9, "pipeline/dataset.py", "Eval golden set from split='eval' holdout", 6, c9)

    pairs = build_preference_pairs(fcon)
    c10 = (
        len(pairs) >= 1
        and all(p["chosen"] and p["rejected"] and p["chosen"] != p["rejected"] for p in pairs)
    )
    grade(10, "pipeline/dataset.py", "DPO pairs (prompt, chosen, rejected) from ok-vs-error", 8, c10)

    clean_pairs = decontaminate(pairs, eval_set)
    held = {e["input"].lower().strip() for e in eval_set}
    c11 = len(clean_pairs) < len(pairs) and all(
        p["prompt"].lower().strip() not in held for p in clean_pairs
    )
    grade(11, "pipeline/dataset.py", "Decontamination drops eval-overlapping pairs", 8, c11)

    pit = point_in_time_features(fcon)
    leaky = naive_leaky_features(fcon)
    m = pit.merge(leaky, on=["user_id", "event_ts"])
    leaks = int((m["spend_leaky"] > m["spend_at_event"]).sum())
    c12 = leaks >= 1 and (m["spend_at_event"] <= m["spend_leaky"]).all()
    grade(12, "pipeline/features.py", "ASOF PIT features; naive join leaks future", 10, c12)
    fcon.close()

    # --- verify.py reproducible ---
    verify_ok, got, total = run_verify()
    c_verify = verify_ok and got == total == 16
    grade(0, "verify.py", f"make verify prints ALL PASS ({got}/{total} checks)", 5, c_verify)

    # --- Print report ---
    print("=" * 72)
    print("DAY 17 LAB — RUBRIC GRADING REPORT (CORE 100 pts)")
    print("=" * 72)
    earned = 0
    for num, where, criterion, pts, ok in scores:
        mark = "PASS" if ok else "FAIL"
        earned += pts if ok else 0
        label = f"#{num}" if num else "—"
        print(f"  [{mark:4}] {label:>3} ({pts:2} pts) {where}")
        print(f"         {criterion}")
    print("-" * 72)
    print(f"  CORE SCORE: {earned}/{CORE_TOTAL}")
    if earned == CORE_TOTAL:
        print("  RESULT: 100/100 — FULL MARKS")
    else:
        failed = [s for s in scores if not s[4]]
        print(f"  RESULT: {earned}/100 — {len(failed)} criterion/criteria FAILED")
        for num, where, criterion, pts, _ in failed:
            print(f"    -> #{num or 'verify'}: {criterion}")

    # Submission artifacts (not in core 100 but required for LMS)
    print("\n" + "=" * 72)
    print("SUBMISSION ARTIFACTS (required for LMS, not counted in 100)")
    print("=" * 72)
    artifacts = [
        ("datasets/eval_golden.jsonl", (ROOT / "datasets/eval_golden.jsonl").exists()),
        ("datasets/preference_pairs.jsonl", (ROOT / "datasets/preference_pairs.jsonl").exists()),
        ("submission/REFLECTION.md", (ROOT / "submission/REFLECTION.md").exists()),
    ]
    for name, ok in artifacts:
        print(f"  [{'OK' if ok else 'MISSING':7}] {name}")

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"DISABLE_PANDERA_IMPORT_WARNING": "True", **dict(__import__("os").environ)},
    )
    pytest_ok = "18 passed" in r.stdout
    print(f"  [{'OK' if pytest_ok else 'FAIL':7}] pytest -q (18 passed)")

    return 0 if earned == CORE_TOTAL else 1


if __name__ == "__main__":
    sys.exit(run_grading())
