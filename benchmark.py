"""Benchmark the Polaris SLM against the validated seeds dataset.

Measures four tiers per example: generated, parses, runs (produces a
DataFrame), matches (structurally equals the reference parquet — column-order
invariant, floats rounded). Reports aggregates overall + by tag + by
difficulty. Saves per-example artifacts (generated code + actual output) to
runs/debug/ so you can `diff` the code or open the parquet in any notebook.

Usage:
  python benchmark.py                                   # full benchmark
  python benchmark.py --limit 3                         # smoke test
  python benchmark.py --oracle                          # harness self-check (100% expected)
  python benchmark.py --constrained                     # Outlines CFG-constrained decoding
  python benchmark.py --debug                           # verbose diff on every failure
  python benchmark.py --out runs/baseline.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.compare import ComparisonResult, compare_dataframes
from dataset.executor import execute_code, load_tpch
from dataset.polars_grammar import build_grammar_gbnf

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"

SYSTEM_PROMPT = """Return only valid Python Polars code (no markdown fences, no prose).

Rules:
- Assign the final DataFrame to `result`.
- Use Polars syntax, NOT pandas: `group_by` (not `groupby`), `pl.col("x")` for columns (never bare strings inside expressions or `.col_name` attribute access).
- Dates: `.dt.year()`, `.dt.month()` for parts; `pl.date(Y, M, D)` for literals; `.is_between(a, b)` for ranges.
- Aggregation: `pl.len()` for row count; use `.alias("name")` to rename outputs.
- The provided DataFrames are already in scope by name — do not recreate them with pl.DataFrame()."""


FEWSHOT: list[tuple[dict, str, str]] = [
    (
        {"sales": {"product": "Utf8", "revenue": "Float64", "region": "Utf8"}},
        "Return all sales in Europe with revenue above 100, sorted by revenue descending.",
        'result = (\n'
        '    sales\n'
        '    .filter((pl.col("region") == "Europe") & (pl.col("revenue") > 100))\n'
        '    .sort("revenue", descending=True)\n'
        ')',
    ),
    (
        {"events": {"user_id": "Int64", "event_type": "Utf8", "ts": "Datetime"}},
        "Count events per event_type for 2024, sorted by count descending.",
        'result = (\n'
        '    events\n'
        '    .filter(pl.col("ts").dt.year() == 2024)\n'
        '    .group_by("event_type")\n'
        '    .agg(pl.len().alias("count"))\n'
        '    .sort("count", descending=True)\n'
        ')',
    ),
    (
        {
            "users": {"user_id": "Int64", "name": "Utf8"},
            "orders": {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"},
        },
        "For each user, return the total amount of their orders, top 5.",
        'result = (\n'
        '    users\n'
        '    .join(orders, on="user_id")\n'
        '    .group_by("name")\n'
        '    .agg(pl.col("amount").sum().alias("total"))\n'
        '    .sort("total", descending=True)\n'
        '    .head(5)\n'
        ')',
    ),
]


def format_user_turn(tables: dict, question: str) -> str:
    return f"Datasets:\n{json.dumps(tables, indent=2)}\n\nQuestion: {question}"


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
        # xgrammar bits, built once on first --constrained call.
        self._xgr = None
        self._xgr_tokinfo = None
        self._xgr_compiler = None

    def _build_prompt(self, message: str, tables: dict) -> str:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_tables, fs_q, fs_a in FEWSHOT:
            messages.append({"role": "user", "content": format_user_turn(fs_tables, fs_q)})
            messages.append({"role": "assistant", "content": fs_a})
        messages.append({"role": "user", "content": format_user_turn(tables, message)})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _ensure_xgrammar(self) -> None:
        """Build the xgrammar tokenizer-info + compiler once and reuse.

        Uses the model's LM-head width as vocab_size (which accounts for
        padding) rather than `tokenizer.vocab_size`, which under-reports for
        several modern tokenizers and causes `token_bitmask.shape` mismatches
        in downstream masking. Handles nested configs (Gemma 4, LFM, etc.)
        which expose vocab_size on a sub-config.
        """
        if self._xgr_compiler is not None:
            return
        import xgrammar as xgr

        vocab_size = self._resolve_vocab_size()
        tokinfo = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=vocab_size
        )
        compiler = xgr.GrammarCompiler(tokinfo)
        # Only commit to self after every step succeeds so a partial init
        # doesn't fool later callers into thinking we're ready.
        self._xgr = xgr
        self._xgr_tokinfo = tokinfo
        self._xgr_compiler = compiler

    def _resolve_vocab_size(self) -> int:
        cfg = self.model.config
        if getattr(cfg, "vocab_size", None):
            return int(cfg.vocab_size)
        for sub in ("text_config", "language_config", "llm_config"):
            sub_cfg = getattr(cfg, sub, None)
            if sub_cfg is not None and getattr(sub_cfg, "vocab_size", None):
                return int(sub_cfg.vocab_size)
        out = self.model.get_output_embeddings()
        if out is not None:
            return int(out.weight.shape[0])
        raise RuntimeError("could not resolve vocab_size from model config")

    @torch.inference_mode()
    def generate(self, message: str, tables: dict, max_new_tokens: int = 512) -> str:
        text = self._build_prompt(message, tables)
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

    @torch.inference_mode()
    def generate_constrained(
        self, message: str, tables: dict, max_new_tokens: int = 512,
    ) -> str:
        """Grammar-constrained decoding driven directly by xgrammar.

        We bypass Outlines' wrapper because it mis-sizes the bitmask with
        LFM2.5-style tokenizers (uses `tokenizer.vocab_size` where xgrammar
        expects `config.vocab_size`). This custom loop owns the KV cache and
        applies the per-step token bitmask from the GrammarMatcher before
        argmax sampling.
        """
        self._ensure_xgrammar()
        xgr = self._xgr
        prompt = self._build_prompt(message, tables)
        grammar_str = build_grammar_gbnf(tables)

        compiled = self._xgr_compiler.compile_grammar(grammar_str)
        matcher = xgr.GrammarMatcher(compiled)
        vocab_size = self._xgr_tokinfo.vocab_size
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model(input_ids=inputs["input_ids"], use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        generated: list[int] = []
        eos = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            if matcher.is_terminated():
                break
            matcher.fill_next_token_bitmask(bitmask)
            xgr.apply_token_bitmask_inplace(logits, bitmask.to(logits.device))
            next_id = int(torch.argmax(logits, dim=-1).item())
            if eos is not None and next_id == eos:
                break
            if not matcher.accept_token(next_id):
                break
            generated.append(next_id)
            step = self.model(
                input_ids=torch.tensor([[next_id]], device=self.model.device),
                past_key_values=past,
                use_cache=True,
            )
            logits = step.logits[:, -1, :]
            past = step.past_key_values

        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_code_fence(response)


def evaluate_one(
    record: dict,
    generated_code: str,
    tables: dict,
    expected: pl.DataFrame | None,
) -> tuple[dict, pl.DataFrame | None]:
    scoped = {name: tables[name] for name in record["tables"] if name in tables}
    prompt_schemas = {name: schema["columns"] for name, schema in record["tables"].items()}

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
    constrained: bool = False,
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

    model = None if oracle else PolarisModel()

    results = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
        prompt_schemas = {name: schema["columns"] for name, schema in rec["tables"].items()}

        t0 = time.time()
        if oracle:
            generated = rec["reference_code"]
            gen_error = None
        else:
            try:
                if constrained:
                    generated = model.generate_constrained(rec["question"], prompt_schemas)
                else:
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
    p.add_argument("--constrained", action="store_true",
                   help="Use grammar-constrained generation (Outlines CFG) with a "
                        "Polars grammar built per-example from the request's schema")
    p.add_argument("--debug", action="store_true",
                   help="Print full diff (code + previews) for each failure")
    p.add_argument("--debug-dir", type=Path, default=Path("runs/debug"),
                   help="Directory for per-example artifacts (generated code + actual parquet)")
    p.add_argument("--no-debug-dir", action="store_true",
                   help="Skip writing per-example artifact files")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save aggregate results JSON")
    args = p.parse_args()

    debug_dir = None if args.no_debug_dir else args.debug_dir

    if args.oracle and args.constrained:
        print("WARN: --constrained has no effect with --oracle; ignoring.")

    results = run(
        args.seeds, args.data, args.expected, debug_dir, args.limit, args.oracle,
        constrained=args.constrained,
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
