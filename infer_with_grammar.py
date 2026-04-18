"""Single-file inference + grammar-constrained generation + retry cascade.

Uses llama_cpp (GGUF) for inference and pygbnf to build the GBNF grammar that
constrains the model to valid Polars method chains over the given schema.

Cascade levels:
  L1 (fast)        — unconstrained greedy; accepted if looks_ok()
  L2 (constrained) — same prompt but grammar-gated via LlamaGrammar
  L3 (retry)       — appends the failing code + feedback, re-generates

After the cascade, mock-execution against empty DataFrames provides one more
round of feedback-driven retries (up to MAX_EXEC_RETRIES).

Install:
    uv pip install llama-cpp-python huggingface-hub pygbnf polars

Run:
    python infer_with_grammar.py
    python infer_with_grammar.py --question "..." --schema '{"t": {"columns": {...}}}'
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field

import pygbnf as cfg
from pygbnf import one_or_more, optional, select, zero_or_more
from pygbnf.nodes import CharacterClass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_FILE = "gemma-4-E2B-it-Q5_K_M.gguf"
MAX_EXEC_RETRIES = 2

SYSTEM_PROMPT = (
    "Return only valid Python Polars code (no markdown fences, no prose).\n\n"
    "Rules:\n"
    "- Assign the final DataFrame to `result`.\n"
    "- The provided DataFrames are already in scope by their table name — do not recreate them.\n"
    "- Use Polars syntax, NOT pandas: `group_by` (not `groupby`), `pl.col(\"x\")` for columns.\n\n"
    "Modern Polars API (use exactly these forms):\n"
    "- df.with_columns(...)          NOT df.with_column(...)\n"
    "- df.sort(\"x\", descending=True) NOT pl.desc(\"x\")\n"
    "- df.group_by(\"x\").agg(...)     NOT df.agg(...) on a DataFrame\n"
    "- expr.str.contains(\"...\")      NOT expr.contains(...)\n"
    "- df.height  or  pl.len()       NOT df.len / df.length\n"
    "- expr.rank(method=\"dense\")     NOT expr.dense(...)\n"
    "- df.filter(pl.col(\"x\") > 10)   NOT df[df[\"x\"] > 10]\n"
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _extract_schema(tables: dict) -> tuple[list[str], list[str]]:
    table_names: list[str] = []
    all_cols: set[str] = set()
    for t, meta in tables.items():
        table_names.append(t)
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        if isinstance(cols, dict):
            all_cols.update(cols.keys())
        else:
            all_cols.update(cols)
    return sorted(set(table_names)), sorted(all_cols)


def _collect_valid_columns(tables: dict) -> set[str]:
    valid: set[str] = set()
    for _name, meta in tables.items():
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        if isinstance(cols, dict):
            valid.update(cols.keys())
        elif isinstance(cols, (list, tuple, set)):
            valid.update(cols)
    return valid


def format_schema_block(tables: dict) -> str:
    lines = ["Datasets:"]
    for name, meta in tables.items():
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        n_rows = meta.get("n_rows") if isinstance(meta, dict) else None
        rows_str = f"  ({n_rows} rows)" if n_rows else ""
        lines.append(f"  {name}{rows_str}")
        if isinstance(cols, dict):
            for col, dtype in cols.items():
                lines.append(f"    {col}: {dtype}")
        else:
            for col in cols:
                lines.append(f"    {col}")
    return "\n".join(lines)


def build_user_prompt(tables: dict, question: str) -> str:
    schema_block = format_schema_block(tables)
    col_names = sorted(_collect_valid_columns(tables))
    col_reminder = f"\nValid column names (use ONLY these): {col_names}" if col_names else ""
    return f"{schema_block}{col_reminder}\n\nTask: {question}"


# ---------------------------------------------------------------------------
# GBNF grammar builder (pygbnf)
# ---------------------------------------------------------------------------

def build_gbnf(tables: dict) -> str:
    """Return a GBNF grammar string restricted to valid Polars chains over tables."""
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
            "[" + ws() + colref()
            + zero_or_more(ws() + "," + ws() + colref())
            + optional(ws() + ",")
            + ws() + "]"
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
        return select(["sum", "mean", "min", "max", "count",
                       "n_unique", "first", "last", "len"])

    @g.rule
    def m_alias():
        return "alias" + ws() + "(" + ws() + string() + ws() + ")"

    @g.rule
    def m_agg():
        return agg_name() + ws() + "(" + ws() + ")"

    @g.rule
    def m_is_between():
        return (
            "is_between" + ws() + "(" + ws()
            + expr() + ws() + "," + ws() + expr()
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
            "rank" + ws() + "(" + ws()
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

    @g.rule
    def col_atom():
        return "pl.col" + ws() + "(" + ws() + colref() + ws() + ")"

    @g.rule
    def pl_len():
        return "pl.len" + ws() + "(" + ws() + ")"

    @g.rule
    def pl_date():
        return (
            "pl.date" + ws() + "(" + ws()
            + int_val() + ws() + "," + ws()
            + int_val() + ws() + "," + ws()
            + int_val() + ws() + ")"
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

    @g.rule
    def unary():
        return select(["-" + ws() + atom_with_methods(), atom_with_methods()])

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

    @g.rule
    def expr_list():
        return (
            "[" + ws() + expr()
            + zero_or_more(ws() + "," + ws() + expr())
            + optional(ws() + ",")
            + ws() + "]"
        )

    @g.rule
    def expr_or_list():
        return select([expr(), expr_list()])

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
            "group_by" + ws() + "(" + ws() + colref_or_list() + ws() + ")"
            + ws() + "." + ws()
            + "agg" + ws() + "(" + ws() + expr_or_list() + ws() + ")"
        )

    @g.rule
    def sort_call():
        bool_list = (
            "[" + ws() + bool_val()
            + zero_or_more(ws() + "," + ws() + bool_val())
            + optional(ws() + ",") + ws() + "]"
        )
        sort_kw = "descending" + ws() + "=" + ws() + select([bool_val(), bool_list])
        sort_key = select([
            colref(),
            "[" + ws() + colref()
            + zero_or_more(ws() + "," + ws() + colref())
            + optional(ws() + ",") + ws() + "]",
        ])
        return (
            "sort" + ws() + "(" + ws()
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
            "rename" + ws() + "(" + ws()
            + "{" + ws()
            + rename_pair()
            + zero_or_more(ws() + "," + ws() + rename_pair())
            + optional(ws() + ",")
            + ws() + "}" + ws() + ")"
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
            "join" + ws() + "(" + ws()
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
        return select(["(" + ws() + pipeline() + ws() + ")", pipeline()])

    @g.rule
    def assign():
        return "result" + ws() + "=" + ws() + rhs()

    g.start("assign")
    return g.to_gbnf()


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

_HALLUCINATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("used `.with_column(` (singular) — use `.with_columns(` instead",
     re.compile(r"\.with_column\s*\(")),
    ("used `pl.desc(` — use `.sort(col, descending=True)` instead",
     re.compile(r"\bpl\.desc\s*\(")),
    ("used `.dense(` — use `.rank(method=\"dense\")` instead",
     re.compile(r"\.dense\s*\(")),
    ("accessed `df.len` as attribute — use `df.height` or `pl.len()`",
     re.compile(r"\.len(?!\s*\()")),
    ("used pandas-style `groupby(` — use `group_by(` in Polars",
     re.compile(r"\bgroupby\s*\(")),
    ("called `.contains(` on Expr — use `.str.contains(` for strings",
     re.compile(r"(?<!\.str)\.contains\s*\(")),
    ("used boolean indexing `df[...>...]` — use `df.filter(pl.col(...) > ...)`",
     re.compile(r"\[\s*\w+\s*\[\s*\"")),
]

_COL_REF = re.compile(r'pl\.col\s*\(\s*"([^"]+)"\s*\)')
_ALIAS = re.compile(r'\.alias\s*\(\s*"([^"]+)"\s*\)')


def detect_hallucinations(code: str) -> list[str]:
    if not code:
        return []
    return [label for label, rx in _HALLUCINATION_PATTERNS if rx.search(code)]


def looks_ok(code: str, tables: dict) -> tuple[bool, str]:
    if not code or not code.strip():
        return False, "empty"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e.msg}"
    if "result =" not in code and "result=" not in code:
        return False, "missing_result_assignment"
    hallucinations = detect_hallucinations(code)
    if hallucinations:
        return False, f"hallucinated_api: {hallucinations[0]}"
    aliases = set(_ALIAS.findall(code))
    valid_cols = _collect_valid_columns(tables) | aliases
    if valid_cols:
        for col in _COL_REF.findall(code):
            if col not in valid_cols:
                return False, f"unknown_column: {col!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Mock execution
# ---------------------------------------------------------------------------

def _resolve_pl_dtype(name: str):
    import polars as pl
    base = name.split("[", 1)[0].split("(", 1)[0].strip()
    return getattr(pl, {"Bool": "Boolean", "String": "Utf8"}.get(base, base), pl.Utf8)


def _build_mock_tables(tables: dict) -> dict:
    import polars as pl
    mock = {}
    for name, meta in tables.items():
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        if isinstance(cols, dict):
            schema = {col: _resolve_pl_dtype(str(dt)) for col, dt in cols.items()}
        elif isinstance(cols, (list, tuple, set)):
            schema = {col: pl.Utf8 for col in cols}
        else:
            continue
        mock[name] = pl.DataFrame(schema=schema)
    return mock


_MISSING_COL_RE = re.compile(r'unable to find column\s+"([^"]+)"', re.IGNORECASE)
_VALID_COLS_RE = re.compile(r'valid columns?:\s*\[([^\]]+)\]', re.IGNORECASE)


def build_structured_feedback(error_msg: str) -> str:
    if not error_msg:
        return "Previous code failed with an unknown error. Rewrite carefully."
    missing = _MISSING_COL_RE.search(error_msg)
    valid = _VALID_COLS_RE.search(error_msg)
    if missing and valid:
        return (
            f'Your previous code referenced column "{missing.group(1)}" which does not exist. '
            f"The ONLY valid column names are: [{valid.group(1)}]. "
            f"Rewrite using EXCLUSIVELY these exact names."
        )
    if "DuplicateError" in error_msg:
        return (
            "Your previous code caused a column-name collision on a join (DuplicateError). "
            "Pre-select/rename the conflicting column before the join, or pass suffix=\"_other\". "
            f"Raw error: {error_msg[:200]}"
        )
    if "ColumnNotFoundError" in error_msg:
        return (
            "Your previous code referenced a column that does not exist. "
            f"Use only the columns listed in the schema. Raw error: {error_msg[:200]}"
        )
    if "SchemaError" in error_msg:
        return (
            "Your previous code raised a SchemaError — likely a type mismatch. "
            f"Raw error: {error_msg[:250]}"
        )
    return f"Your previous code failed with: {error_msg[:400]}. Rewrite carefully."


def try_mock_execute(code: str, tables: dict) -> tuple[bool, str]:
    if not code or not code.strip():
        return False, "empty code"
    try:
        import polars as pl
        mock_tables = _build_mock_tables(tables)
    except Exception as e:
        return True, f"mock_setup_failed: {e}"
    env: dict = {"pl": pl, **mock_tables}
    try:
        exec(code, env)  # noqa: S102
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:400]}"
    if "result" not in env:
        return False, "code did not assign 'result'"
    res = env["result"]
    try:
        if hasattr(res, "collect"):
            res.collect()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:400]}"
    return True, ""


# ---------------------------------------------------------------------------
# llama_cpp model wrapper
# ---------------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class LlamaCppModel:
    """Thin wrapper around llama_cpp.Llama exposing generate / generate_constrained
    / generate_with_feedback — same interface as GemmaModel in gemma_model.py."""

    def __init__(self, repo_id: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                 n_ctx: int = 8192, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        print(f"Loading {repo_id} / {filename} ...", flush=True)
        self._llm = Llama.from_pretrained(
            repo_id=repo_id, filename=filename,
            n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False,
        )
        print("Model ready.", flush=True)

    def _chat(self, messages: list[dict], grammar=None,
              max_tokens: int = 512, temperature: float = 0.0) -> str:
        out = self._llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return strip_code_fence(out["choices"][0]["message"]["content"])

    def _base_messages(self, question: str, tables: dict) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(tables, question)},
        ]

    def generate(self, question: str, tables: dict, max_tokens: int = 512) -> str:
        return self._chat(self._base_messages(question, tables), max_tokens=max_tokens)

    def generate_constrained(self, question: str, tables: dict,
                             max_tokens: int = 512) -> str:
        from llama_cpp import LlamaGrammar
        gbnf = build_gbnf(tables)
        grammar = LlamaGrammar.from_string(gbnf)
        return self._chat(self._base_messages(question, tables),
                          grammar=grammar, max_tokens=max_tokens)

    def generate_with_feedback(self, question: str, tables: dict,
                               previous_code: str, feedback: str,
                               max_tokens: int = 512) -> str:
        messages = self._base_messages(question, tables)
        messages.append({"role": "assistant", "content": previous_code or ""})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous code has the following issues: {feedback}. "
                "Rewrite using only the modern Polars APIs. "
                "Assign the final DataFrame to `result`. Return only corrected code, no prose."
            ),
        })
        return self._chat(messages, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    code: str
    level: str          # fast | constrained | retry | exec_retry<n> | fallback
    reason: str
    l1_code: str = field(default="")
    l1_reason: str = field(default="")


def _log(msg: str) -> None:
    print(f"[cascade] {msg}", flush=True)


def run_cascade(model: LlamaCppModel, question: str, tables: dict,
                disable_constrained: bool = False) -> CascadeResult:
    # L1 — fast
    code_v1 = model.generate(question, tables)
    ok, reason = looks_ok(code_v1, tables)
    if ok:
        return CascadeResult(code=code_v1, level="fast", reason="ok",
                             l1_code=code_v1, l1_reason="ok")
    l1_code, l1_reason = code_v1, reason
    _log(f"L1 rejected: {l1_reason}")

    # L2 — constrained (pygbnf + LlamaGrammar)
    code_v2: str | None = None
    if not disable_constrained:
        try:
            _log("trying constrained generation (pygbnf GBNF)...")
            code_v2 = model.generate_constrained(question, tables)
            ok2, reason2 = looks_ok(code_v2, tables)
            if ok2:
                return CascadeResult(code=code_v2, level="constrained", reason="ok",
                                     l1_code=l1_code, l1_reason=l1_reason)
            _log(f"L2 rejected: {reason2}")
        except Exception as e:
            _log(f"L2 raised {type(e).__name__}: {e}")
            code_v2 = None

    # L3 — retry with feedback
    hallucinations = detect_hallucinations(l1_code)
    feedback = "; ".join(hallucinations) if hallucinations else l1_reason
    try:
        _log(f"L3 retry, feedback: {feedback[:120]}")
        code_v3 = model.generate_with_feedback(question, tables, l1_code, feedback)
        ok3, _ = looks_ok(code_v3, tables)
        if ok3:
            return CascadeResult(code=code_v3, level="retry", reason="ok",
                                 l1_code=l1_code, l1_reason=l1_reason)
    except Exception as e:
        _log(f"L3 raised {type(e).__name__}: {e}")
        code_v3 = ""

    best = code_v2 or l1_code or code_v3 or (
        f"result = {next(iter(tables))}" if tables else "result = None"
    )
    return CascadeResult(code=best, level="fallback", reason=l1_reason,
                         l1_code=l1_code, l1_reason=l1_reason)


def run_cascade_with_exec_retry(model: LlamaCppModel, question: str, tables: dict,
                                disable_constrained: bool = False,
                                max_retries: int = MAX_EXEC_RETRIES) -> CascadeResult:
    result = run_cascade(model, question, tables, disable_constrained=disable_constrained)

    for attempt in range(1, max_retries + 1):
        ok, err = try_mock_execute(result.code, tables)
        if ok:
            return result
        _log(f"mock_exec failed (attempt {attempt}): {err}")
        feedback = build_structured_feedback(err)
        try:
            new_code = model.generate_with_feedback(question, tables, result.code, feedback)
        except Exception as e:
            _log(f"exec-retry gen raised {type(e).__name__}: {e}")
            break
        result = CascadeResult(
            code=new_code,
            level=f"exec_retry{attempt}",
            reason=f"fixed_from_mock_exec: {err[:100]}",
            l1_code=result.l1_code,
            l1_reason=result.l1_reason,
        )

    ok, err = try_mock_execute(result.code, tables)
    if not ok:
        _log(f"mock_exec still failing after {max_retries} retries: {err}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_DEFAULT_SCHEMA = {
    "customer": {
        "columns": {
            "c_custkey": "Int64",
            "c_name": "Utf8",
            "c_mktsegment": "Utf8",
            "c_acctbal": "Float64",
        }
    }
}
_DEFAULT_QUESTION = (
    'From `customer`, keep rows where c_mktsegment == "BUILDING", '
    "then select c_custkey and c_name, sorted by c_acctbal descending."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grammar-constrained Polars inference + retry")
    parser.add_argument("--question", default=_DEFAULT_QUESTION)
    parser.add_argument("--schema", default=None,
                        help="JSON string, e.g. '{\"t\": {\"columns\": {\"col\": \"Int64\"}}}'")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--no-constrained", action="store_true")
    parser.add_argument("--max-retries", type=int, default=MAX_EXEC_RETRIES)
    args = parser.parse_args()

    tables = json.loads(args.schema) if args.schema else _DEFAULT_SCHEMA

    try:
        model = LlamaCppModel(repo_id=args.repo, filename=args.file)
    except ImportError:
        print("llama-cpp-python is not installed. Run:")
        print("  uv pip install llama-cpp-python huggingface-hub")
        return 1

    print(f"\nQuestion : {args.question}")
    print(f"Schema   : {json.dumps(tables, indent=2)}\n")

    result = run_cascade_with_exec_retry(
        model, args.question, tables,
        disable_constrained=args.no_constrained,
        max_retries=args.max_retries,
    )

    print(f"\n--- result (level={result.level}, reason={result.reason!r}) ---")
    print(result.code)
    print("---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
