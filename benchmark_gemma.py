"""Benchmark Gemma 4 E2B against the validated seeds dataset.

Fork of benchmark.py adapted for google/gemma-4-E2B-it. Keeps the same 4-tier
scoring (generated / parses / runs / matches via compare_dataframes) and the
same chat-format few-shot approach, so results are directly comparable to the
LFM2 baseline in benchmark.py.

Differences vs benchmark.py:
- MODEL_NAME = "google/gemma-4-E2B-it"
- GemmaModel uses AutoProcessor (multimodal-aware) + enable_thinking=False
- Enriched SYSTEM_PROMPT covering the API hallucinations observed on the baseline
  (with_column, pl.desc, .dense, .str.contains, df[bool], .len)
- Extra few-shot covering the window/rank pattern (weakest tag for Gemma baseline)
- Schema injected to the model includes n_rows per table (A3)

Usage:
  python benchmark_gemma.py --oracle --limit 3   # validate harness
  python benchmark_gemma.py --limit 3            # smoke test with Gemma
  python benchmark_gemma.py                      # full benchmark
  python benchmark_gemma.py --out runs/gemma_promptv2.json
"""
import argparse
import ast
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from dataset.compare import ComparisonResult, compare_dataframes
from dataset.executor import execute_code, load_tpch
from gemma_prompt import FEWSHOT, SYSTEM_PROMPT, format_user_turn

MODEL_NAME = "google/gemma-4-E2B-it"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class GemmaModel:
    def __init__(self, name: str = MODEL_NAME):
        print(f"Loading {name}...")
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.eos_token_id = self.processor.tokenizer.eos_token_id

    @torch.inference_mode()
    def generate(self, message: str, tables: dict, max_new_tokens: int = 512) -> str:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_tables, fs_q, fs_a in FEWSHOT:
            messages.append({"role": "user", "content": format_user_turn(fs_tables, fs_q)})
            messages.append({"role": "assistant", "content": fs_a})
        messages.append({"role": "user", "content": format_user_turn(tables, message)})
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
        )
        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return strip_code_fence(response)


def evaluate_one(
    record: dict,
    generated_code: str,
    tables: dict,
    expected: pl.DataFrame | None,
) -> tuple[dict, pl.DataFrame | None]:
    scoped = {name: tables[name] for name in record["tables"] if name in tables}
    prompt_schemas = {
        name: {"columns": schema["columns"], "n_rows": schema.get("n_rows")}
        for name, schema in record["tables"].items()
    }

    out: dict = {
        "id": record["id"],
        "tags": record["tags"],
        "difficulty": record["difficulty"],
        "prompt_schemas": prompt_schemas,
        "generated_code": generated_code,
        "parses": False,
        "runs": False,
        "matches": False,
        "comparison": None,
        "error": None,
    }

    if not generated_code.strip():
        out["error"] = "empty generation"
        return out, None

    try:
        ast.parse(generated_code)
        out["parses"] = True
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e}"
        return out, None

    exec_result = execute_code(generated_code, scoped)
    if not exec_result.success:
        out["error"] = exec_result.error
        return out, None
    out["runs"] = True

    if expected is None:
        out["error"] = "no expected parquet for this record"
        return out, exec_result.result

    cmp: ComparisonResult = compare_dataframes(exec_result.result, expected)
    out["matches"] = cmp.match
    out["comparison"] = asdict(cmp)
    if not cmp.match:
        out["error"] = cmp.summary()
    return out, exec_result.result


def _stage(r: dict) -> str:
    if r["matches"]:
        return "match"
    if r["runs"]:
        cmp = r.get("comparison") or {}
        return cmp.get("reason", "wrong-output")
    if r["parses"]:
        return "crash"
    return "syntax"


def report(results: list[dict], debug: bool = False) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    parses = sum(r["parses"] for r in results)
    runs = sum(r["runs"] for r in results)
    matches = sum(r["matches"] for r in results)
    avg_latency = sum(r.get("latency_sec", 0.0) for r in results) / n

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS — Gemma 4 E2B  ({n} examples)")
    print(f"{'=' * 70}\n")
    print(f"  parses:      {parses:>3d}/{n}  ({parses / n * 100:5.1f}%)")
    print(f"  runs:        {runs:>3d}/{n}  ({runs / n * 100:5.1f}%)")
    print(f"  matches:     {matches:>3d}/{n}  ({matches / n * 100:5.1f}%)")
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
    for stage in ("syntax", "crash", "empty_actual", "column_set", "row_count", "content"):
        if stage_counts.get(stage):
            print(f"    {stage:15s}  {stage_counts[stage]}")

    failures = [r for r in results if not r["matches"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            stage = _stage(r)
            print(f"    [{stage:12s}] {r['id']:35s}  {r.get('error', '')}")

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
    print("\n>>> generated code:")
    print(r["generated_code"] or "<empty>")
    cmp = r.get("comparison")
    if cmp:
        print(f"\n>>> columns — actual: {cmp['actual_columns']}")
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
        raise SystemExit(f"No TPC-H data in {data_dir}. Run `python -m dataset.gen_tpch` first.")

    expected = load_expected(expected_dir)
    if not expected:
        print(
            f"WARN: no expected parquets in {expected_dir}. "
            "Run `python -m dataset.build_seeds` to (re)produce them."
        )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model = None if oracle else GemmaModel()

    results = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
        prompt_schemas = {
            name: {"columns": schema["columns"], "n_rows": schema.get("n_rows")}
            for name, schema in rec["tables"].items()
        }

        t0 = time.time()
        if oracle:
            generated = rec["reference_code"]
            gen_error = None
        else:
            try:
                generated = model.generate(rec["question"], prompt_schemas)
                gen_error = None
            except Exception as e:
                generated = ""
                gen_error = str(e)
                print(f"GEN ERROR: {e}")
        latency = time.time() - t0

        if gen_error:
            result = {
                "id": rec["id"],
                "tags": rec["tags"],
                "difficulty": rec["difficulty"],
                "prompt_schemas": prompt_schemas,
                "generated_code": "",
                "parses": False,
                "runs": False,
                "matches": False,
                "comparison": None,
                "error": f"generation failed: {gen_error}",
                "latency_sec": latency,
            }
            results.append(result)
            continue

        result, actual_df = evaluate_one(rec, generated, tables, expected.get(rec["id"]))
        result["latency_sec"] = latency
        results.append(result)

        if debug_dir is not None:
            (debug_dir / f"{rec['id']}.generated.py").write_text(generated or "")
            if actual_df is not None:
                actual_df.write_parquet(debug_dir / f"{rec['id']}.actual.parquet")

        flag = (
            "OK   " if result["matches"]
            else "RAN  " if result["runs"]
            else "PARSE" if result["parses"]
            else "FAIL "
        )
        print(f"{flag}  {latency:5.2f}s")

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=Path, default=Path("data/seeds.jsonl"))
    p.add_argument("--data", type=Path, default=Path("data/tpch"))
    p.add_argument("--expected", type=Path, default=Path("data/expected"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--oracle", action="store_true",
                   help="Use reference_code as generation (validates the harness)")
    p.add_argument("--debug", action="store_true",
                   help="Print full diff (code + previews) for each failure")
    p.add_argument("--debug-dir", type=Path, default=Path("runs/debug_gemma"),
                   help="Directory for per-example artifacts (generated code + actual parquet)")
    p.add_argument("--no-debug-dir", action="store_true",
                   help="Skip writing per-example artifact files")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save aggregate results JSON")
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
