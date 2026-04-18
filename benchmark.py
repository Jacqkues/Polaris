"""Benchmark the Polaris SLM against the validated seeds dataset.

Measures four tiers of success per example: generated, parses, runs (produces
a DataFrame), matches (output hash equals reference). Reports aggregates
overall, by tag, and by difficulty; saves full results to JSON for before/after
comparison across training runs.

Usage:
  python benchmark.py                                   # full benchmark
  python benchmark.py --limit 3                         # smoke test
  python benchmark.py --oracle                          # validate harness (100% expected)
  python benchmark.py --out runs/baseline.json
"""
import argparse
import ast
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.executor import execute_code, load_tpch
from dataset.hashing import hash_dataframe

MODEL_NAME = "LiquidAI/LFM2-8B-A1B"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class PolarisModel:
    def __init__(self, name: str = MODEL_NAME):
        print(f"Loading {name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, message: str, tables: dict, max_new_tokens: int = 256) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only valid Python Polars code. "
                    "No markdown fences. "
                    "Assign the final Polars DataFrame to result. "
                    f"Available datasets: {json.dumps(tables, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": message},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return strip_code_fence(response)


def evaluate_one(record: dict, generated_code: str, tables: dict) -> dict:
    scoped = {name: tables[name] for name in record["tables"] if name in tables}
    prompt_schemas = {name: schema["columns"] for name, schema in record["tables"].items()}

    out = {
        "id": record["id"],
        "tags": record["tags"],
        "difficulty": record["difficulty"],
        "prompt_schemas": prompt_schemas,
        "generated_code": generated_code,
        "parses": False,
        "runs": False,
        "matches": False,
        "error": None,
    }

    if not generated_code.strip():
        out["error"] = "empty generation"
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

    actual_hash = hash_dataframe(exec_result.result)
    out["matches"] = actual_hash == record["expected_output_hash"]
    out["actual_n_rows"] = exec_result.result.height
    out["actual_columns"] = exec_result.result.columns
    if not out["matches"]:
        out["error"] = (
            f"output mismatch: got {exec_result.result.height} rows, "
            f"expected {record['expected_n_rows']}"
        )
    return out


def report(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    parses = sum(r["parses"] for r in results)
    runs = sum(r["runs"] for r in results)
    matches = sum(r["matches"] for r in results)
    avg_latency = sum(r.get("latency_sec", 0.0) for r in results) / n

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS  ({n} examples)")
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

    failures = [r for r in results if not r["matches"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            stage = (
                "wrong-output" if r["runs"]
                else "crash" if r["parses"]
                else "syntax"
            )
            print(f"    [{stage:12s}] {r['id']:35s}  {r.get('error', '')}")
    print()


def run(seeds_path: Path, data_dir: Path, limit: int | None, oracle: bool) -> list[dict]:
    records = [
        json.loads(line) for line in seeds_path.read_text().splitlines() if line.strip()
    ]
    if limit:
        records = records[:limit]

    tables = load_tpch(data_dir)
    if not tables:
        raise SystemExit(f"No TPC-H data in {data_dir}. Run `python -m dataset.gen_tpch` first.")

    model = None if oracle else PolarisModel()

    results = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
        prompt_schemas = {name: schema["columns"] for name, schema in rec["tables"].items()}

        t0 = time.time()
        if oracle:
            generated = rec["reference_code"]
        else:
            try:
                generated = model.generate(rec["question"], prompt_schemas)
            except Exception as e:
                generated = ""
                print(f"GEN ERROR: {e}")
                latency = time.time() - t0
                result = {
                    "id": rec["id"],
                    "tags": rec["tags"],
                    "difficulty": rec["difficulty"],
                    "prompt_schemas": prompt_schemas,
                    "generated_code": "",
                    "parses": False,
                    "runs": False,
                    "matches": False,
                    "error": f"generation failed: {e}",
                    "latency_sec": latency,
                }
                results.append(result)
                continue
        latency = time.time() - t0

        result = evaluate_one(rec, generated, tables)
        result["latency_sec"] = latency
        results.append(result)

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
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--oracle", action="store_true",
                   help="Use reference_code as generation (validates the harness)")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save results JSON")
    args = p.parse_args()

    results = run(args.seeds, args.data, args.limit, args.oracle)
    report(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved results → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
