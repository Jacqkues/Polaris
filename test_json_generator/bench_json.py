"""Benchmark the JSON-based query generator against the seeds dataset.

Same 4-tier scoring as benchmark_gemma.py (generated / parses / runs / matches)
plus two extra tiers specific to the JSON path:

    plan_valid   — the model output parsed into a valid QueryPlan
    converted    — json_to_polars() produced non-empty Polars code

Results include the raw JSON plan so you can inspect what the model
actually planned vs. what it was converted to.

Usage:
  # Validate harness (no model)
  python bench_json.py --oracle --limit 3

  # Full benchmark
  python bench_json.py

  # Smoke test with first 5 examples
  python bench_json.py --limit 5

  # Save results
  python bench_json.py --out runs/bench_json.json
"""
from __future__ import annotations

import ast
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

# Allow running from inside the test_json_generator subdirectory
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import polars as pl

from dataset.compare import ComparisonResult, compare_dataframes
from dataset.executor import execute_code, load_tpch
from test_json_generator.json_query_generator import JsonQueryGenerator, QueryPlan, json_to_polars


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    record: dict,
    generated_code: str,
    tables: dict[str, pl.DataFrame],
    expected: pl.DataFrame | None,
    json_plan: dict | None,
) -> dict:
    scoped = {name: tables[name] for name in record["tables"] if name in tables}

    out: dict = {
        "id": record["id"],
        "tags": record["tags"],
        "difficulty": record["difficulty"],
        "json_plan": json_plan,
        "generated_code": generated_code,
        "plan_valid": json_plan is not None,
        "converted": bool(generated_code and generated_code.strip()),
        "parses": False,
        "runs": False,
        "matches": False,
        "comparison": None,
        "error": None,
    }

    if not generated_code or not generated_code.strip():
        out["error"] = "empty conversion"
        return out

    try:
        ast.parse(generated_code)
        out["parses"] = True
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e}"
        return out

    exec_result = execute_code(generated_code, scoped)
    if not exec_result.success:
        out["error"] = exec_result.error
        return out
    out["runs"] = True

    if expected is None:
        out["error"] = "no expected parquet for this record"
        return out

    cmp: ComparisonResult = compare_dataframes(exec_result.result, expected)
    out["matches"] = cmp.match
    out["comparison"] = asdict(cmp)
    if not cmp.match:
        out["error"] = cmp.summary()
    return out


def _stage(r: dict) -> str:
    if r["matches"]:
        return "match"
    if r["runs"]:
        cmp = r.get("comparison") or {}
        return cmp.get("reason", "wrong-output")
    if r["parses"]:
        return "crash"
    if r["converted"]:
        return "syntax"
    if r["plan_valid"]:
        return "convert-fail"
    return "invalid-plan"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(results: list[dict], debug: bool = False) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    plan_valid = sum(r["plan_valid"] for r in results)
    converted = sum(r["converted"] for r in results)
    parses = sum(r["parses"] for r in results)
    runs = sum(r["runs"] for r in results)
    matches = sum(r["matches"] for r in results)
    avg_latency = sum(r.get("latency_sec", 0.0) for r in results) / n

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS — JSON Generator + Gemma 4 E2B  ({n} examples)")
    print(f"{'=' * 70}\n")
    print(f"  plan valid:  {plan_valid:>3d}/{n}  ({plan_valid / n * 100:5.1f}%)  JSON parsed by QueryPlan schema")
    print(f"  converted:   {converted:>3d}/{n}  ({converted / n * 100:5.1f}%)  json_to_polars produced code")
    print(f"  parses:      {parses:>3d}/{n}  ({parses / n * 100:5.1f}%)  AST-valid Python")
    print(f"  runs:        {runs:>3d}/{n}  ({runs / n * 100:5.1f}%)  executes without error")
    print(f"  matches:     {matches:>3d}/{n}  ({matches / n * 100:5.1f}%)  output matches reference")
    print(f"  avg latency: {avg_latency:.2f}s")

    tag_hits: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for tag in r["tags"]:
            tag_hits[tag].append(r["matches"])
    print("\n  By tag:")
    for tag in sorted(tag_hits):
        hits = tag_hits[tag]
        print(
            f"    {tag:15s}  {sum(hits):>2d}/{len(hits):<2d}  "
            f"{sum(hits) / len(hits) * 100:5.1f}%"
        )

    diff_hits: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        diff_hits[r["difficulty"]].append(r["matches"])
    print("\n  By difficulty:")
    for d in sorted(diff_hits):
        hits = diff_hits[d]
        print(
            f"    d{d}              {sum(hits):>2d}/{len(hits):<2d}  "
            f"{sum(hits) / len(hits) * 100:5.1f}%"
        )

    stage_counts: dict[str, int] = defaultdict(int)
    for r in results:
        stage_counts[_stage(r)] += 1
    print("\n  Failure stages:")
    for stage in ("invalid-plan", "convert-fail", "syntax", "crash",
                  "empty_actual", "column_set", "row_count", "content"):
        if stage_counts.get(stage):
            print(f"    {stage:15s}  {stage_counts[stage]}")

    failures = [r for r in results if not r["matches"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            stage = _stage(r)
            print(f"    [{stage:14s}] {r['id']:35s}  {r.get('error', '')}")

    if debug and failures:
        print(f"\n{'=' * 70}")
        print("DEBUG DETAILS")
        print(f"{'=' * 70}")
        for r in failures:
            _print_debug(r)
    print()


def _print_debug(r: dict) -> None:
    print(f"\n--- {r['id']}  [{_stage(r)}] ---")
    print(f"tags: {r['tags']}  difficulty: d{r['difficulty']}")
    print(f"error: {r.get('error')}")
    if r.get("json_plan"):
        print("\n>>> JSON plan:")
        print(json.dumps(r["json_plan"], indent=2))
    print("\n>>> generated code:")
    print(r["generated_code"] or "<empty>")
    cmp = r.get("comparison")
    if cmp:
        print(f"\n>>> columns — actual:   {cmp['actual_columns']}")
        print(f">>> columns — expected: {cmp['expected_columns']}")
        if cmp.get("missing_columns"):
            print(f">>> missing: {cmp['missing_columns']}")
        if cmp.get("extra_columns"):
            print(f">>> extra:   {cmp['extra_columns']}")
        print(f"\n>>> rows — actual: {cmp['actual_n_rows']}  expected: {cmp['expected_n_rows']}")
        print("\n>>> actual preview:")
        print(cmp["actual_preview"])
        print("\n>>> expected preview:")
        print(cmp["expected_preview"])
        if cmp.get("detail"):
            print(f"\n>>> detail: {cmp['detail']}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_expected(expected_dir: Path) -> dict[str, pl.DataFrame]:
    if not expected_dir.exists():
        return {}
    return {p.stem: pl.read_parquet(p) for p in expected_dir.glob("*.parquet")}


def run(
    seeds_path: Path,
    data_dir: Path,
    expected_dir: Path,
    debug_dir: Path | None,
    limit: int | None,
    oracle: bool,
) -> list[dict]:
    records = [
        json.loads(line) for line in seeds_path.read_text().splitlines() if line.strip()
    ]
    if limit:
        records = records[:limit]

    tables = load_tpch(data_dir)
    if not tables:
        raise SystemExit(
            f"No TPC-H data in {data_dir}. Run `python -m dataset.gen_tpch` first."
        )

    expected = load_expected(expected_dir)
    if not expected:
        print(
            f"WARN: no expected parquets in {expected_dir}. "
            "Run `python -m dataset.build_seeds` to (re)produce them."
        )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model: JsonQueryGenerator | None = None if oracle else JsonQueryGenerator()

    results = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
        prompt_schemas = {
            name: {"columns": schema["columns"], "n_rows": schema.get("n_rows")}
            for name, schema in rec["tables"].items()
        }

        t0 = time.time()
        json_plan: dict | None = None
        generated = ""
        gen_error: str | None = None

        if oracle:
            generated = rec["reference_code"]
        else:
            try:
                plan: QueryPlan = model.generate(prompt_schemas, rec["question"])
                json_plan = plan.model_dump()
                generated = json_to_polars(plan)
            except Exception as e:
                gen_error = str(e)
                print(f"GEN ERROR: {e}")

        latency = time.time() - t0

        if gen_error:
            results.append({
                "id": rec["id"],
                "tags": rec["tags"],
                "difficulty": rec["difficulty"],
                "json_plan": None,
                "generated_code": "",
                "plan_valid": False,
                "converted": False,
                "parses": False,
                "runs": False,
                "matches": False,
                "comparison": None,
                "error": f"generation failed: {gen_error}",
                "latency_sec": latency,
            })
            print("FAIL ")
            continue

        result = evaluate_one(
            rec, generated, tables, expected.get(rec["id"]), json_plan
        )
        result["latency_sec"] = latency
        results.append(result)

        if debug_dir is not None:
            (debug_dir / f"{rec['id']}.plan.json").write_text(
                json.dumps(json_plan, indent=2) if json_plan else "{}"
            )
            (debug_dir / f"{rec['id']}.generated.py").write_text(generated or "")

        flag = (
            "OK   " if result["matches"]
            else "RAN  " if result["runs"]
            else "PARSE" if result["parses"]
            else "FAIL "
        )
        print(f"{flag}  {latency:5.2f}s")

    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=Path, default=Path("data/seeds.jsonl"))
    p.add_argument("--data", type=Path, default=Path("data/tpch"))
    p.add_argument("--expected", type=Path, default=Path("data/expected"))
    p.add_argument("--limit", type=int, default=None, help="Run only the first N examples")
    p.add_argument("--oracle", action="store_true",
                   help="Use reference_code as generation (validates the harness, no model needed)")
    p.add_argument("--debug", action="store_true",
                   help="Print full diff (code + previews) for each failure")
    p.add_argument("--debug-dir", type=Path, default=Path("runs/debug_json"),
                   help="Directory for per-example artifacts (.plan.json + .generated.py)")
    p.add_argument("--no-debug-dir", action="store_true",
                   help="Skip writing per-example artifact files")
    p.add_argument("--out", type=Path, default=None,
                   help="Save aggregate results to this JSON file")
    args = p.parse_args()

    debug_dir = None if args.no_debug_dir else args.debug_dir

    results = run(
        args.seeds, args.data, args.expected, debug_dir, args.limit, args.oracle
    )
    report(results, debug=args.debug)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved results → {args.out}")
    if debug_dir is not None:
        print(f"Per-example artifacts → {debug_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
