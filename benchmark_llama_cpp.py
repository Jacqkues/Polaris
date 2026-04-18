"""Benchmark a GGUF Liquid LFM2 via llama.cpp with grammar-constrained decoding.

Mirrors `benchmark.py` but swaps the HF transformers + Outlines stack for
`llama-cpp-python`. Grammar is the full Polars GBNF from
`dataset.polars_grammar.build_grammar_gbnf`, specialized per-example with the
request's table/column schema so the decoder can only emit valid chains over
real identifiers.

Usage:
  python benchmark_llama_cpp.py                          # full benchmark, constrained
  python benchmark_llama_cpp.py --limit 3                # smoke
  python benchmark_llama_cpp.py --unconstrained          # no grammar (for A/B)
  python benchmark_llama_cpp.py --repo LiquidAI/... --file model.gguf
  LFM2_GGUF=/path/to.gguf python benchmark_llama_cpp.py  # local file

Install:
  uv pip install llama-cpp-python huggingface-hub
  # CUDA:
  #   uv pip install llama-cpp-python \
  #     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmark import (
    FEWSHOT,
    SYSTEM_PROMPT,
    evaluate_one,
    format_user_turn,
    load_expected,
    report,
    strip_code_fence,
)
from dataset.executor import load_tpch
from dataset.polars_grammar import build_grammar_gbnf

DEFAULT_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_FILE = "gemma-4-E2B-it-BF16.gguf"


def _flatten_gbnf(grammar: str) -> str:
    """Join rule-continuation lines into a single physical line.

    `build_grammar_gbnf` (shared with xgrammar) wraps long alternatives like
        methcall ::= a | b | c
                   | d | e
    onto multiple lines. xgrammar accepts that; llama.cpp's GBNF parser does
    not — it treats a newline as end-of-rule and then chokes on the leading
    `|` of the next line ("expecting name at |..."). We merge any line whose
    first non-whitespace char is `|` into the previous line.
    """
    out: list[str] = []
    for line in grammar.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("|") and out:
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(line)
    return "\n".join(out)


class LlamaCppModel:
    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILE,
        local_path: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ):
        from llama_cpp import Llama

        if local_path:
            print(f"Loading local GGUF: {local_path}")
            self.llm = Llama(
                model_path=local_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        else:
            print(f"Loading {repo} / {filename}")
            self.llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        # Cache compiled grammars per-schema — GBNF compilation is non-trivial
        # and every TPC-H example reuses one of a few table combinations.
        self._grammar_cache: dict[tuple[str, ...], object] = {}

    def _messages(self, message: str, tables: dict) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_tables, fs_q, fs_a in FEWSHOT:
            msgs.append({"role": "user", "content": format_user_turn(fs_tables, fs_q)})
            msgs.append({"role": "assistant", "content": fs_a})
        msgs.append({"role": "user", "content": format_user_turn(tables, message)})
        return msgs

    def _grammar_for(self, tables: dict):
        from llama_cpp import LlamaGrammar

        key = tuple(sorted(tables.keys()))
        cached = self._grammar_cache.get(key)
        if cached is not None:
            return cached
        gbnf = _flatten_gbnf(build_grammar_gbnf(tables))
        grammar = LlamaGrammar.from_string(gbnf, verbose=False)
        self._grammar_cache[key] = grammar
        return grammar

    def generate(
        self,
        message: str,
        tables: dict,
        max_new_tokens: int = 512,
        constrained: bool = True,
    ) -> str:
        kwargs: dict = {
            "messages": self._messages(message, tables),
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if constrained:
            kwargs["grammar"] = self._grammar_for(tables)
        out = self.llm.create_chat_completion(**kwargs)
        text = out["choices"][0]["message"]["content"]
        return strip_code_fence(text)


def run(
    seeds_path: Path,
    data_dir: Path,
    expected_dir: Path,
    debug_dir: Path | None,
    limit: int | None,
    constrained: bool,
    repo: str,
    filename: str,
    local_path: str | None,
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

    model = LlamaCppModel(repo=repo, filename=filename, local_path=local_path)

    results: list[dict] = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
        prompt_schemas = {name: schema["columns"] for name, schema in rec["tables"].items()}

        t0 = time.time()
        try:
            generated = model.generate(
                rec["question"], prompt_schemas, constrained=constrained
            )
            gen_error = None
        except Exception as e:
            generated = ""
            gen_error = str(e)
            print(f"GEN ERROR: {e}")
        latency = time.time() - t0

        if gen_error:
            results.append({
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
            })
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
    p.add_argument("--unconstrained", action="store_true",
                   help="Disable grammar constraint (for A/B comparison)")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help="HF GGUF repo id (ignored if --local or LFM2_GGUF is set)")
    p.add_argument("--file", dest="filename", default=DEFAULT_FILE,
                   help="GGUF filename inside the repo")
    p.add_argument("--local", default=os.environ.get("LFM2_GGUF"),
                   help="Path to a local GGUF file (overrides --repo/--file)")
    p.add_argument("--debug", action="store_true",
                   help="Print full diff (code + previews) for each failure")
    p.add_argument("--debug-dir", type=Path, default=Path("runs/debug_llama_cpp"))
    p.add_argument("--no-debug-dir", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    debug_dir = None if args.no_debug_dir else args.debug_dir

    results = run(
        args.seeds,
        args.data,
        args.expected,
        debug_dir,
        args.limit,
        constrained=not args.unconstrained,
        repo=args.repo,
        filename=args.filename,
        local_path=args.local,
    )
    report(results, debug=args.debug)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved results -> {args.out}")
    if debug_dir is not None:
        print(f"Per-example artifacts -> {debug_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
