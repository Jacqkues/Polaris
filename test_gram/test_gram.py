"""Grammar-constrained Polars generation using pygbnf + llama.cpp (gemma-4-E2B-it).

pygbnf builds a GBNF grammar that restricts the model to valid Polars method
chains over the given schema. llama_cpp applies that grammar during inference.

Requires:
    uv pip install pygbnf llama-cpp-python huggingface-hub

Run:
    python -m test_gram.test_gram

Env vars:
    GEMMA_REPO  — HF repo  (default: unsloth/gemma-4-E2B-it-GGUF)
    GEMMA_FILE  — filename  (default: gemma-4-E2B-it-Q5_K_M.gguf)
    GEMMA_GGUF  — local file path (overrides repo/file)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pygbnf as cfg
from pygbnf import one_or_more, optional, select, zero_or_more
from pygbnf.nodes import CharacterClass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.polars_grammar import validate as lark_validate  # noqa: E402

# ---------------------------------------------------------------------------
# Schema used for the test
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
    'Task: from `customer`, keep rows where c_mktsegment == "BUILDING", '
    "then select c_custkey and c_name."
)

DEFAULT_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_FILE = "gemma-4-E2B-it-Q5_K_M.gguf"


# ---------------------------------------------------------------------------
# Grammar builder — pygbnf DSL
# ---------------------------------------------------------------------------

def _extract_schema(tables: dict) -> tuple[list[str], list[str]]:
    table_names: list[str] = []
    all_cols: set[str] = set()
    for t, meta in tables.items():
        table_names.append(t)
        if isinstance(meta, dict):
            cols = meta.get("columns", meta)
        else:
            cols = meta
        if isinstance(cols, dict):
            all_cols.update(cols.keys())
        else:
            all_cols.update(cols)
    return sorted(set(table_names)), sorted(all_cols)


def build_gbnf(tables: dict) -> str:
    """Return a GBNF grammar string built with pygbnf for the given schema."""
    table_names, col_names = _extract_schema(tables)
    g = cfg.Grammar()

    @g.rule
    def ws():
        return zero_or_more(select(" \t\n\r"))

    @g.rule
    def tablename():
        return select(table_names)

    @g.rule
    def colstr():
        return select([f'"{c}"' for c in col_names])

    @g.rule
    def identstr():
        return (
            '"'
            + CharacterClass("a-zA-Z_")
            + zero_or_more(CharacterClass("a-zA-Z0-9_"))
            + '"'
        )

    @g.rule
    def colref():
        return select([colstr(), identstr()])

    @g.rule
    def colref_list():
        return (
            "["
            + ws()
            + colref()
            + zero_or_more(ws() + "," + ws() + colref())
            + optional(ws() + ",")
            + ws()
            + "]"
        )

    @g.rule
    def colref_or_list():
        return select([colref(), colref_list()])

    @g.rule
    def bool_val():
        return select(["True", "False"])

    @g.rule
    def string():
        return '"' + zero_or_more(CharacterClass('^"')) + '"'

    @g.rule
    def int_val():
        return one_or_more(select("0123456789"))

    @g.rule
    def float_val():
        return (
            one_or_more(select("0123456789"))
            + "."
            + one_or_more(select("0123456789"))
        )

    @g.rule
    def pl_type():
        return select([
            "pl.Int64", "pl.Int32", "pl.Float64", "pl.Float32",
            "pl.Utf8", "pl.String", "pl.Date", "pl.Datetime", "pl.Boolean",
        ])

    @g.rule
    def agg_name():
        return select([
            "sum", "mean", "min", "max", "count",
            "n_unique", "first", "last", "len",
        ])

    # -- expression methods --------------------------------------------------

    @g.rule
    def m_alias():
        return "alias" + ws() + "(" + ws() + string() + ws() + ")"

    @g.rule
    def m_agg():
        return agg_name() + ws() + "(" + ws() + ")"

    @g.rule
    def m_is_between():
        return (
            "is_between"
            + ws() + "(" + ws()
            + expr()
            + ws() + "," + ws()
            + expr()
            + ws() + ")"
        )

    @g.rule
    def rank_kw():
        return select([
            "method" + ws() + "=" + ws() + string(),
            "descending" + ws() + "=" + ws() + bool_val(),
        ])

    @g.rule
    def m_rank():
        return (
            "rank"
            + ws() + "(" + ws()
            + optional(rank_kw() + zero_or_more(ws() + "," + ws() + rank_kw()))
            + ws() + ")"
        )

    @g.rule
    def m_over():
        return "over" + ws() + "(" + ws() + colref_or_list() + ws() + ")"

    @g.rule
    def m_cast():
        return "cast" + ws() + "(" + ws() + pl_type() + ws() + ")"

    @g.rule
    def str_meth():
        return select([
            "contains" + ws() + "(" + ws() + string() + ws() + ")",
            "starts_with" + ws() + "(" + ws() + string() + ws() + ")",
            "ends_with" + ws() + "(" + ws() + string() + ws() + ")",
            "to_lowercase" + ws() + "(" + ws() + ")",
            "to_uppercase" + ws() + "(" + ws() + ")",
            "len_chars" + ws() + "(" + ws() + ")",
        ])

    @g.rule
    def m_str():
        return "str" + ws() + "." + ws() + str_meth()

    @g.rule
    def dt_meth():
        return select([
            "year" + ws() + "(" + ws() + ")",
            "month" + ws() + "(" + ws() + ")",
            "day" + ws() + "(" + ws() + ")",
        ])

    @g.rule
    def m_dt():
        return "dt" + ws() + "." + ws() + dt_meth()

    @g.rule
    def method_call():
        return select([m_alias(), m_agg(), m_is_between(), m_rank(),
                       m_over(), m_cast(), m_str(), m_dt()])

    @g.rule
    def expr_method():
        return "." + ws() + method_call()

    # -- atoms ---------------------------------------------------------------

    @g.rule
    def col_atom():
        return "pl.col" + ws() + "(" + ws() + colref() + ws() + ")"

    @g.rule
    def pl_len():
        return "pl.len" + ws() + "(" + ws() + ")"

    @g.rule
    def pl_date():
        return (
            "pl.date"
            + ws() + "(" + ws()
            + int_val() + ws() + "," + ws()
            + int_val() + ws() + "," + ws()
            + int_val()
            + ws() + ")"
        )

    @g.rule
    def paren_expr():
        return "(" + ws() + expr() + ws() + ")"

    @g.rule
    def expr_atom():
        return select([col_atom(), pl_len(), pl_date(), paren_expr()])

    @g.rule
    def literal_atom():
        return select([float_val(), int_val(), colstr(), identstr(), bool_val()])

    @g.rule
    def atom_with_methods():
        return select([
            expr_atom() + zero_or_more(expr_method()),
            literal_atom(),
        ])

    # -- expression hierarchy ------------------------------------------------

    @g.rule
    def unary():
        return select([
            "-" + ws() + atom_with_methods(),
            atom_with_methods(),
        ])

    @g.rule
    def prod_op():
        return select(["*", "/"])

    @g.rule
    def prod_expr():
        return unary() + zero_or_more(ws() + prod_op() + ws() + unary())

    @g.rule
    def sum_op():
        return select(["+", "-"])

    @g.rule
    def sum_expr():
        return prod_expr() + zero_or_more(ws() + sum_op() + ws() + prod_expr())

    @g.rule
    def cmp_op():
        return select(["==", "!=", "<=", ">=", "<", ">"])

    @g.rule
    def cmp_rhs():
        return select([sum_expr(), string()])

    @g.rule
    def cmp_expr():
        return sum_expr() + optional(ws() + cmp_op() + ws() + cmp_rhs())

    @g.rule
    def not_expr():
        return select(["~" + ws() + not_expr(), cmp_expr()])

    @g.rule
    def and_expr():
        return not_expr() + zero_or_more(ws() + "&" + ws() + not_expr())

    @g.rule
    def expr():
        return and_expr() + zero_or_more(ws() + "|" + ws() + and_expr())

    # -- expression lists ----------------------------------------------------

    @g.rule
    def expr_list():
        return (
            "["
            + ws()
            + expr()
            + zero_or_more(ws() + "," + ws() + expr())
            + optional(ws() + ",")
            + ws()
            + "]"
        )

    @g.rule
    def expr_or_list():
        return select([expr(), expr_list()])

    # -- method calls --------------------------------------------------------

    @g.rule
    def filter_call():
        return "filter" + ws() + "(" + ws() + expr() + ws() + ")"

    @g.rule
    def select_call():
        return "select" + ws() + "(" + ws() + expr_or_list() + ws() + ")"

    @g.rule
    def with_cols_call():
        return "with_columns" + ws() + "(" + ws() + expr_or_list() + ws() + ")"

    @g.rule
    def group_by_agg_call():
        return (
            "group_by"
            + ws() + "(" + ws() + colref_or_list() + ws() + ")"
            + ws() + "." + ws()
            + "agg" + ws() + "(" + ws() + expr_or_list() + ws() + ")"
        )

    @g.rule
    def sort_call():
        bool_list = (
            "["
            + ws()
            + bool_val()
            + zero_or_more(ws() + "," + ws() + bool_val())
            + optional(ws() + ",")
            + ws()
            + "]"
        )
        sort_kw = "descending" + ws() + "=" + ws() + select([bool_val(), bool_list])
        sort_key = select([
            colref(),
            "[" + ws() + colref() + zero_or_more(ws() + "," + ws() + colref()) + optional(ws() + ",") + ws() + "]",
        ])
        return (
            "sort"
            + ws() + "(" + ws()
            + sort_key
            + optional(ws() + "," + ws() + sort_kw)
            + ws() + ")"
        )

    @g.rule
    def head_call():
        return "head" + ws() + "(" + ws() + int_val() + ws() + ")"

    @g.rule
    def limit_call():
        return "limit" + ws() + "(" + ws() + int_val() + ws() + ")"

    @g.rule
    def unique_call():
        return "unique" + ws() + "(" + ws() + optional(colref_or_list()) + ws() + ")"

    @g.rule
    def rename_pair():
        return colref() + ws() + ":" + ws() + string()

    @g.rule
    def rename_call():
        return (
            "rename"
            + ws() + "(" + ws()
            + "{" + ws()
            + rename_pair()
            + zero_or_more(ws() + "," + ws() + rename_pair())
            + optional(ws() + ",")
            + ws() + "}"
            + ws() + ")"
        )

    @g.rule
    def join_kw():
        return select([
            "left_on" + ws() + "=" + ws() + colref(),
            "right_on" + ws() + "=" + ws() + colref(),
            "on" + ws() + "=" + ws() + colref_or_list(),
            "how" + ws() + "=" + ws() + string(),
        ])

    @g.rule
    def join_call():
        return (
            "join"
            + ws() + "(" + ws()
            + tablename()
            + one_or_more(ws() + "," + ws() + join_kw())
            + ws() + ")"
        )

    @g.rule
    def meth_call():
        return select([
            filter_call(), select_call(), with_cols_call(), group_by_agg_call(),
            join_call(), sort_call(), head_call(), limit_call(),
            unique_call(), rename_call(),
        ])

    @g.rule
    def chain_method():
        return "." + ws() + meth_call()

    @g.rule
    def pipeline():
        return tablename() + one_or_more(ws() + chain_method())

    @g.rule
    def rhs():
        return select([
            "(" + ws() + pipeline() + ws() + ")",
            pipeline(),
        ])

    @g.rule
    def assign():
        return "result" + ws() + "=" + ws() + rhs()

    g.start("assign")
    return g.to_gbnf()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_llm():
    from llama_cpp import Llama

    local = os.environ.get("GEMMA_GGUF")
    if local:
        return Llama(model_path=local, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    repo = os.environ.get("GEMMA_REPO", DEFAULT_REPO)
    fname = os.environ.get("GEMMA_FILE", DEFAULT_FILE)
    return Llama.from_pretrained(
        repo_id=repo, filename=fname, n_ctx=4096, n_gpu_layers=-1, verbose=False
    )


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def run() -> int:
    try:
        from llama_cpp import LlamaGrammar
    except ImportError:
        print("llama-cpp-python is not installed. Run:")
        print('  uv pip install llama-cpp-python huggingface-hub')
        return 1

    print("Building GBNF grammar with pygbnf...")
    gbnf = build_gbnf(SCHEMA)
    print(f"Grammar preview (first 400 chars):\n{gbnf[:400]}\n{'...' if len(gbnf) > 400 else ''}")

    grammar = LlamaGrammar.from_string(gbnf)
    print("LlamaGrammar compiled OK.")

    print(f"\nLoading model ({DEFAULT_REPO} / {DEFAULT_FILE})...")
    llm = _load_llm()

    messages = [
        {
            "role": "system",
            "content": "You output a single Polars expression assigned to `result`.",
        },
        {"role": "user", "content": PROMPT},
    ]

    print("Running constrained inference...")
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
