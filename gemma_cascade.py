"""Cascade generation orchestrator for Gemma 4 E2B.

Three-level waterfall that adapts to per-request difficulty:

    L1 (fast)        → GemmaModel.generate(...)
                       Rejected if looks_ok() fails (AST error, hallucinated
                       API, unknown column, etc.) → fall through to L2.

    L2 (constrained) → GemmaModel.generate_constrained(...)  [OPTIONAL]
                       Skipped entirely if `llguidance`/`outlines` not
                       installed, or if disabled via flag/env. The cascade
                       stays useful with just L1 + L3 as a fast→retry loop.

    L3 (retry)       → GemmaModel.generate_with_feedback(...)
                       Re-prompts with the failing L1 code + an explicit
                       feedback message derived from detect_hallucinations().

    Fallback         → returns the "least bad" candidate (L2 > L1) to avoid
                       an empty response on the server.

Validation is purely STATIC (AST + regex + column check against the schema):
this keeps the logic identical between the FastAPI server (main_cascade.py,
no tables at hand) and the local benchmark (benchmark_gemma.py), so local
scores reliably predict production behavior.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Availability probe — run once at import time
# ---------------------------------------------------------------------------

def _probe_constrained_backend() -> bool:
    """Returns True iff Outlines + llguidance can be imported. Cached lazily."""
    try:
        import outlines  # noqa: F401
        import llguidance  # noqa: F401
        return True
    except ImportError:
        return False


CONSTRAINED_AVAILABLE = _probe_constrained_backend()


def constrained_disabled_by_env() -> bool:
    return os.environ.get("POLARIS_DISABLE_CONSTRAINED", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------

# Each entry: (label, compiled regex). The label is what we'll pass back
# to the model in the L3 feedback turn, so it must be self-explanatory.
_HALLUCINATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("used `.with_column(` (singular) — use `.with_columns(` instead",
     re.compile(r"\.with_column\s*\(")),
    ("used `pl.desc(` — it does not exist, use `.sort(col, descending=True)`",
     re.compile(r"\bpl\.desc\s*\(")),
    ("used `.dense(` — use `.rank(method=\"dense\")` instead",
     re.compile(r"\.dense\s*\(")),
    ("accessed `df.len` as an attribute — use `df.height` or `pl.len()`",
     re.compile(r"\.len(?!\s*\()")),
    ("used pandas-style `groupby(` — use `group_by(` (underscore) in Polars",
     re.compile(r"\bgroupby\s*\(")),
    ("called `.contains(` on an Expr — use `.str.contains(` for strings",
     re.compile(r"(?<!\.str)\.contains\s*\(")),
    ("used boolean indexing `df[...>...]` — use `df.filter(pl.col(...) > ...)`",
     re.compile(r"\[\s*\w+\s*\[\s*\"")),
]


def detect_hallucinations(code: str) -> list[str]:
    """Return a list of human-readable labels for every hallucinated pattern
    found in `code`. Empty list = code is clean on the regex checks."""
    if not code:
        return []
    return [label for label, rx in _HALLUCINATION_PATTERNS if rx.search(code)]


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

_COL_REF = re.compile(r'pl\.col\s*\(\s*"([^"]+)"\s*\)')
_ALIAS = re.compile(r'\.alias\s*\(\s*"([^"]+)"\s*\)')


def _collect_valid_columns(tables: dict) -> set[str]:
    """Gather column names across all tables, tolerating all three formats:
    - enriched : {name: {"columns": {col: type}, "n_rows": n}}
    - flat legacy : {name: {col: type}}
    - list legacy : {name: [col, ...]}
    """
    valid: set[str] = set()
    for _name, meta in tables.items():
        if isinstance(meta, dict):
            cols = meta.get("columns", meta)  # prefer "columns" key, else meta itself
        else:
            cols = meta  # meta is probably a list/tuple/set
        if isinstance(cols, dict):
            valid.update(cols.keys())
        elif isinstance(cols, (list, tuple, set)):
            valid.update(cols)
    return valid


def looks_ok(code: str, tables: dict) -> tuple[bool, str]:
    """Static validation of generated Polars code.

    Returns (True, "ok") or (False, reason). Safe to call without a live
    Polars runtime, so usable both in the FastAPI server and the benchmark.
    """
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

    # Columns referenced via pl.col("X") must be in the schema OR be an alias
    # introduced earlier in the same chain.
    aliases = set(_ALIAS.findall(code))
    valid_cols = _collect_valid_columns(tables) | aliases
    if valid_cols:  # only check if we actually have a schema to validate against
        for col in _COL_REF.findall(code):
            if col not in valid_cols:
                return False, f"unknown_column: {col!r}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    code: str
    level: str  # "fast" | "constrained" | "retry" | "fallback"
    reason: str  # why the final level was reached
    l1_code: str = ""  # for debug: the initial attempt (may be empty)
    l1_reason: str = ""  # why L1 was rejected, if it was


def run_cascade(
    model,
    question: str,
    tables: dict,
    *,
    disable_constrained: bool = False,
) -> CascadeResult:
    """Run the fast → constrained → retry cascade and return the best code.

    `model` must expose `generate`, `generate_constrained`, and
    `generate_with_feedback` — i.e. a GemmaModel instance or a compatible
    mock (used in tests).

    `disable_constrained=True` forces skipping L2 even if the backend is
    available (useful for A/B: isolate the effect of retry alone).
    """
    # -- Level 1 — fast -----------------------------------------------------
    code_v1 = model.generate(question, tables)
    ok, reason = looks_ok(code_v1, tables)
    if ok:
        return CascadeResult(code=code_v1, level="fast", reason="ok",
                             l1_code=code_v1, l1_reason="ok")
    l1_reason = reason
    l1_code = code_v1

    # -- Level 2 — constrained (optional) -----------------------------------
    code_v2: str | None = None
    skip_l2 = (
        disable_constrained
        or constrained_disabled_by_env()
        or not CONSTRAINED_AVAILABLE
    )
    if not skip_l2:
        try:
            code_v2 = model.generate_constrained(question, tables)
            ok2, reason2 = looks_ok(code_v2, tables)
            if ok2:
                return CascadeResult(code=code_v2, level="constrained", reason="ok",
                                     l1_code=l1_code, l1_reason=l1_reason)
        except Exception as e:
            # Grammar compile error, dep issue, CUDA OOM, etc.
            # Don't crash the cascade — fall through.
            code_v2 = None
            _log_skip(f"constrained level raised {type(e).__name__}: {e}")

    # -- Level 3 — retry with feedback --------------------------------------
    hallucinations = detect_hallucinations(l1_code)
    # Build a feedback string. If no specific hallucination was detected
    # (e.g. unknown column), fall back to the validator's generic reason.
    if hallucinations:
        feedback = "; ".join(hallucinations)
    else:
        feedback = l1_reason
    try:
        code_v3 = model.generate_with_feedback(question, tables, l1_code, feedback)
        ok3, reason3 = looks_ok(code_v3, tables)
        if ok3:
            return CascadeResult(code=code_v3, level="retry", reason="ok",
                                 l1_code=l1_code, l1_reason=l1_reason)
    except Exception as e:
        _log_skip(f"retry level raised {type(e).__name__}: {e}")
        code_v3 = ""

    # -- Fallback — prefer ANY candidate that passes static validation,
    # otherwise fall back to a safe default. We must NEVER return code that
    # fails looks_ok (we already know it does) just because it's non-empty —
    # e.g. the model may have produced ")" or a syntax-broken fragment. The
    # safe default (`result = <first_table>`) at least parses, runs, and
    # returns *something* so polars.bench can continue.
    for candidate in (code_v2, code_v3, l1_code):
        if candidate:
            ok, _ = looks_ok(candidate, tables)
            if ok:
                return CascadeResult(
                    code=candidate, level="fallback_ok",
                    reason=l1_reason, l1_code=l1_code, l1_reason=l1_reason,
                )
    return CascadeResult(
        code=_safe_default(tables), level="fallback_default",
        reason=l1_reason, l1_code=l1_code, l1_reason=l1_reason,
    )


def _safe_default(tables: dict) -> str:
    """Last-resort code when everything failed. Returns the first table
    so the server never responds with an empty string."""
    if tables:
        first = next(iter(tables))
        return f"result = {first}"
    return "result = None"


def _log_skip(msg: str) -> None:
    """Thin wrapper around print so tests can monkeypatch if needed."""
    print(f"[cascade] {msg}")


# ---------------------------------------------------------------------------
# Mock execution — run the generated code against empty DataFrames
# ---------------------------------------------------------------------------

_PL_DTYPE_MAP = {
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
    "Boolean", "Bool",
    "Utf8", "String", "Categorical",
    "Date", "Datetime", "Time", "Duration",
    "List", "Object", "Null",
}


def _resolve_pl_dtype(name: str):
    """Map a Polars type string (as found in seeds) to a polars.DataType.
    Falls back to Utf8 for unknown types so mock exec still works."""
    import polars as pl
    # Handle parameterized types like "Datetime[μs]", "List[Int64]"
    base = name.split("[", 1)[0].split("(", 1)[0].strip()
    # Common aliases
    aliases = {"Bool": "Boolean", "String": "Utf8"}
    base = aliases.get(base, base)
    return getattr(pl, base, pl.Utf8)


def _build_mock_tables(tables_schema: dict) -> dict:
    """Create empty Polars DataFrames matching the provided schema. Used for
    mock-exec validation — enough for ColumnNotFoundError, DuplicateError on
    joins, SchemaError on cast, etc. Insufficient for data-dependent errors."""
    import polars as pl
    mock = {}
    for name, meta in tables_schema.items():
        cols = meta.get("columns", meta) if isinstance(meta, dict) else meta
        if isinstance(cols, dict):
            pl_schema = {col: _resolve_pl_dtype(str(dtype)) for col, dtype in cols.items()}
        elif isinstance(cols, (list, tuple, set)):
            pl_schema = {col: pl.Utf8 for col in cols}
        else:
            continue
        mock[name] = pl.DataFrame(schema=pl_schema)
    return mock


_MISSING_COL_RE = re.compile(r'unable to find column\s+"([^"]+)"', re.IGNORECASE)
_VALID_COLS_RE = re.compile(r'valid columns?:\s*\[([^\]]+)\]', re.IGNORECASE)


def build_structured_feedback(error_msg: str) -> str:
    """Turn a raw Polars error into an actionable feedback message for the LLM.

    Recognizes common patterns and reformulates them with an explicit
    corrective instruction. Falls back to a generic wording when no known
    pattern matches. The goal is to maximize the signal the model sees —
    a parsed "valid columns: [...]" is far easier to act on than a raw
    traceback.
    """
    if not error_msg:
        return "Previous code failed with an unknown error. Rewrite carefully."

    missing = _MISSING_COL_RE.search(error_msg)
    valid = _VALID_COLS_RE.search(error_msg)
    if missing and valid:
        return (
            f'Your previous code referenced column "{missing.group(1)}" '
            f"which does not exist. "
            f"The ONLY valid column names are: [{valid.group(1)}]. "
            f"Rewrite using EXCLUSIVELY these exact names — do not invent or paraphrase."
        )
    if "DuplicateError" in error_msg:
        return (
            "Your previous code caused a column-name collision on a join "
            "(DuplicateError). Either pre-select/rename the conflicting column "
            "before the join, or pass suffix=\"_other\" to .join(...) to "
            f"disambiguate. Raw error: {error_msg[:200]}"
        )
    if "ColumnNotFoundError" in error_msg:
        return (
            "Your previous code referenced a column that does not exist in "
            "the schema. Use only the columns listed in the Datasets block above. "
            f"Raw error: {error_msg[:200]}"
        )
    if "SchemaError" in error_msg:
        return (
            "Your previous code raised a SchemaError — likely a type mismatch "
            "(e.g. comparing a List column to a scalar, or casting to a "
            "wrong type). Revisit the types in the Datasets block above. "
            f"Raw error: {error_msg[:250]}"
        )
    if "AttributeError" in error_msg:
        return (
            "Your previous code called an attribute or method that doesn't "
            "exist on the object. Review the Modern Polars API rules in the "
            f"system prompt. Raw error: {error_msg[:250]}"
        )
    return f"Your previous code failed with: {error_msg[:400]}. Rewrite carefully."


def try_mock_execute(code: str, tables_schema: dict) -> tuple[bool, str]:
    """Execute the generated code against empty DataFrames matching the schema.

    Returns (success, error_message). On success, error_message is "".
    Catches ColumnNotFoundError, DuplicateError, SchemaError, AttributeError,
    SyntaxError, and anything else that fails at plan-build time. Does NOT
    catch data-dependent issues (wrong aggregation, output shape mismatch…).
    """
    if not code or not code.strip():
        return False, "empty code"
    try:
        import polars as pl
        mock_tables = _build_mock_tables(tables_schema)
    except Exception as e:
        # If we can't even build the mock schema, skip mock exec (don't block)
        return True, f"mock_setup_failed: {type(e).__name__}: {e}"
    env: dict = {"pl": pl, **mock_tables}
    try:
        exec(code, env)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:400]}"
    if "result" not in env:
        return False, "code did not assign 'result'"
    # Force plan evaluation if it's a LazyFrame
    res = env["result"]
    try:
        if hasattr(res, "collect"):
            res.collect()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:400]}"
    return True, ""


def run_cascade_with_exec_retry(
    model,
    question: str,
    tables: dict,
    *,
    disable_constrained: bool = False,
    max_retries: int = 2,
) -> CascadeResult:
    """Cascade + post-validation via mock execution against empty DataFrames.

    After the base cascade returns a code, we run it against empty DFs matching
    the schema. If it raises (ColumnNotFoundError, DuplicateError on join,
    SchemaError on cast…), we re-prompt the model with the actual error
    message — up to `max_retries` times. Stops as soon as mock exec passes.

    The error message from Polars is usually very informative (it lists the
    valid columns on ColumnNotFoundError, suggests using a `suffix` kwarg on
    join duplicates…), so the retry has strong leverage.
    """
    result = run_cascade(model, question, tables, disable_constrained=disable_constrained)

    for attempt in range(1, max_retries + 1):
        ok, err = try_mock_execute(result.code, tables)
        if ok:
            return result
        _log_skip(f"mock_exec failed (attempt {attempt}): {err}")
        # Re-prompt with a structured feedback derived from the Polars error
        # (parses missing-column names + valid-columns list for max clarity).
        feedback = build_structured_feedback(err)
        try:
            new_code = model.generate_with_feedback(
                question, tables, result.code, feedback
            )
        except Exception as e:
            _log_skip(f"exec-retry gen raised {type(e).__name__}: {e}")
            break
        # Keep a level label that makes the source of this code obvious
        new_level = f"exec_retry{attempt}"
        # Update result; mark reason with the last fixed error
        result = CascadeResult(
            code=new_code,
            level=new_level,
            reason=f"fixed_from_mock_exec: {err[:100]}",
            l1_code=result.l1_code,
            l1_reason=result.l1_reason,
        )

    # Final mock-exec check — if it's STILL failing after retries, we return
    # a safe default rather than the last broken attempt. Returning broken
    # code ("result = None" at worst) is strictly better than returning
    # something that won't even parse (e.g. a lone ")" from a degenerate
    # retry) and causes polars.bench to SyntaxError on our output.
    ok, err = try_mock_execute(result.code, tables)
    if not ok:
        _log_skip(f"mock_exec still failing after {max_retries} retries: {err}")
        # Also try looks_ok on the current code — if it passes static but
        # fails mock exec, it's still better than a safe default (partial
        # credit possible). If it fails both, safe default is the floor.
        static_ok, _ = looks_ok(result.code, tables)
        if not static_ok:
            return CascadeResult(
                code=_safe_default(tables),
                level="exec_retry_giveup",
                reason=f"all_retries_failed: {err[:100]}",
                l1_code=result.l1_code,
                l1_reason=result.l1_reason,
            )
    return result
