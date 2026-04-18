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

    # -- Fallback — best-effort, never empty --------------------------------
    best = code_v2 or l1_code or code_v3 or _safe_default(tables)
    return CascadeResult(code=best, level="fallback", reason=l1_reason,
                         l1_code=l1_code, l1_reason=l1_reason)


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
