"""Grammar-constrained Polars generation with a GGUF Liquid LFM2 via llama.cpp.

Requires:
    uv pip install "llama-cpp-python[server]" huggingface-hub

Run:
    python -m tests.test_llama_grammar

`Llama.from_pretrained` downloads the GGUF from the Hub on first run and
caches it under ~/.cache/huggingface/hub. Override with env vars:
    LFM2_REPO=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
    LFM2_FILE=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
Or point at a local file with LFM2_GGUF=/path/to/file.gguf.

The test:
  1. Loads a tiny TPC-H-ish schema.
  2. Builds a GBNF grammar (llama.cpp's grammar format) that only emits
     Polars method chains over the given tables/columns.
  3. Asks the model for Polars code, constrained by that grammar.
  4. Validates the result against the project's Lark grammar.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.polars_grammar import validate as lark_validate  # noqa: E402


# ---------------------------------------------------------------------------
# GBNF builder (llama.cpp grammar format)
# ---------------------------------------------------------------------------
# GBNF is llama.cpp's own grammar dialect. Roughly:
#   rule ::= alt1 | alt2
#   "literal"       - a quoted literal
#   [a-z]           - a char class
#   rule*, rule+, rule? - quantifiers
# See: llama.cpp/grammars/README.md

GBNF_BASE = r"""
root ::= "result = (\n    " pipeline "\n)"
pipeline ::= table-ref (nl method)+
nl ::= "\n    "
table-ref ::= tablename
method ::= "." meth-call
meth-call ::= filter-call | select-call | with-cols-call | group-by-call | agg-call | sort-call | head-call | limit-call | unique-call
filter-call ::= "filter(" expr ")"
select-call ::= "select(" expr-or-list ")"
with-cols-call ::= "with_columns(" expr-or-list ")"
agg-call ::= "agg(" expr-or-list ")"
group-by-call ::= "group_by(" colref-or-list ")"
sort-call ::= "sort(" colref-or-list ("," ws "descending=" bool)? ")"
head-call ::= "head(" int ")"
limit-call ::= "limit(" int ")"
unique-call ::= "unique(" colref-or-list? ")"
colref-or-list ::= colstr | "[" colstr ("," ws colstr)* "]"
expr-or-list ::= expr | "[" expr ("," ws expr)* "]"
expr ::= or-expr
or-expr ::= and-expr (ws "|" ws and-expr)*
and-expr ::= cmp-expr (ws "&" ws cmp-expr)*
cmp-expr ::= sum-expr (ws cmp-op ws sum-expr)?
cmp-op ::= "==" | "!=" | "<=" | ">=" | "<" | ">"
sum-expr ::= prod-expr (ws sum-op ws prod-expr)*
sum-op ::= "+" | "-"
prod-expr ::= atom-m (ws prod-op ws atom-m)*
prod-op ::= "*" | "/"
atom-m ::= atom ("." atom-meth)*
atom ::= col-atom | pl-len | string | float | int | colstr | bool | "(" expr ")"
col-atom ::= "pl.col(" colstr ")"
pl-len ::= "pl.len()"
atom-meth ::= alias-m | agg-m | strns-m | dtns-m
alias-m ::= "alias(" string ")"
agg-m ::= ("sum" | "mean" | "min" | "max" | "count" | "n_unique" | "first" | "last") "()"
strns-m ::= "str." ("contains(" string ")" | "starts_with(" string ")" | "ends_with(" string ")" | "to_lowercase()" | "to_uppercase()")
dtns-m ::= "dt." ("year()" | "month()" | "day()")
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt]
int ::= [0-9]+
float ::= [0-9]+ "." [0-9]+
bool ::= "True" | "False"
ws ::= " "?
%TABLENAME%
%COLSTR%
"""


def _gbnf_literal(s: str) -> str:
    # GBNF strings use C-style escapes; table/column names are plain idents here.
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_gbnf(tables: dict) -> str:
    """Specialize the base GBNF with one alternative per table/column."""
    table_names: list[str] = []
    all_cols: set[str] = set()
    for t, meta in tables.items():
        table_names.append(t)
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        all_cols.update(cols.keys() if isinstance(cols, dict) else cols)

    table_names = sorted(set(table_names))
    col_names = sorted(all_cols)

    tablename_rule = "tablename ::= " + " | ".join(_gbnf_literal(t) for t in table_names)
    # colstr is the *quoted* column form that appears inside `pl.col("...")`.
    colstr_rule = "colstr ::= " + " | ".join(_gbnf_literal(f'"{c}"') for c in col_names)

    return (
        GBNF_BASE.replace("%TABLENAME%", tablename_rule).replace("%COLSTR%", colstr_rule)
    )


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

SCHEMA = {
    "customer": {
        "columns": {
            "c_custkey": "Int64",
            "c_name": "Utf8",
            "c_mktsegment": "Utf8",
            "c_acctbal": "Float64",
        }
    }
}

PROMPT = (
    "Return only Polars code (no fences, no comments). "
    "Assign the final DataFrame to `result`. "
    f"Schema: {json.dumps(SCHEMA)}.\n"
    "Task: from `customer`, keep rows where c_mktsegment == \"BUILDING\", "
    "then select c_custkey and c_name."
)


DEFAULT_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_FILE = "gemma-4-E2B-it-Q5_K_M.gguf"


def _load_llm():
    from llama_cpp import Llama

    local = os.environ.get("LFM2_GGUF")
    if local:
        return Llama(
            model_path=local,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

    repo = os.environ.get("LFM2_REPO", DEFAULT_REPO)
    fname = os.environ.get("LFM2_FILE", DEFAULT_FILE)
    return Llama.from_pretrained(
        repo_id=repo,
        filename=fname,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )


def run() -> int:
    try:
        from llama_cpp import LlamaGrammar
    except ImportError:
        print("llama-cpp-python is not installed. Run:")
        print('  uv pip install "llama-cpp-python" huggingface-hub')
        return 1

    gbnf = build_gbnf(SCHEMA)
    grammar = LlamaGrammar.from_string(gbnf)
    llm = _load_llm()

    messages = [
        {
            "role": "system",
            "content": "You output a single Polars expression assigned to `result`.",
        },
        {"role": "user", "content": PROMPT},
    ]

    out = llm.create_chat_completion(
        messages=messages,
        grammar=grammar,
        max_tokens=256,
        temperature=0.0,
    )
    text = out["choices"][0]["message"]["content"].strip()
    print("\n--- generated ---\n" + text + "\n-----------------")

    ok, err = lark_validate(text, SCHEMA)
    if ok:
        print("OK: output parses against the project Lark grammar.")
        return 0
    print(f"FAIL: Lark rejected the output: {err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(run())
