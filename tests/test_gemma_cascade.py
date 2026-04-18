"""Tests for gemma_cascade.py — static validator + cascade orchestrator.

All tests run WITHOUT loading the Gemma model (use a mock). Safe to run
locally on a machine without GPU or `llguidance`.

Run:
    python -m tests.test_gemma_cascade
    python -m pytest tests/test_gemma_cascade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gemma_cascade import (  # noqa: E402
    CascadeResult,
    _collect_valid_columns,
    build_structured_feedback,
    detect_hallucinations,
    looks_ok,
    run_cascade,
    run_cascade_with_exec_retry,
    try_mock_execute,
)


# ---------------------------------------------------------------------------
# Helpers: mock GemmaModel for run_cascade tests
# ---------------------------------------------------------------------------

class MockModel:
    """Minimal mock of GemmaModel. Each generate method returns a canned
    response and records how often it was called."""

    def __init__(
        self,
        fast: str = "",
        constrained: str = "",
        retry: str = "",
        *,
        constrained_raises: Exception | None = None,
        retry_raises: Exception | None = None,
    ) -> None:
        self._fast = fast
        self._constrained = constrained
        self._retry = retry
        self._constrained_raises = constrained_raises
        self._retry_raises = retry_raises
        self.calls: list[str] = []

    def generate(self, message: str, tables: dict) -> str:
        self.calls.append("fast")
        return self._fast

    def generate_constrained(self, message: str, tables: dict) -> str:
        self.calls.append("constrained")
        if self._constrained_raises is not None:
            raise self._constrained_raises
        return self._constrained

    def generate_with_feedback(
        self, message: str, tables: dict, previous_code: str, feedback: str
    ) -> str:
        self.calls.append("retry")
        if self._retry_raises is not None:
            raise self._retry_raises
        return self._retry


TPCH_TABLES = {
    "customer": {
        "columns": {"c_custkey": "Int64", "c_name": "Utf8", "c_nationkey": "Int64"},
        "n_rows": 1500,
    },
    "orders": {
        "columns": {"o_orderkey": "Int64", "o_custkey": "Int64", "o_totalprice": "Float64"},
        "n_rows": 15000,
    },
}


GOOD_CODE = 'result = customer.filter(pl.col("c_custkey") > 10).select("c_name")'


# ---------------------------------------------------------------------------
# looks_ok — nominal
# ---------------------------------------------------------------------------

def test_looks_ok_accepts_modern_polars() -> None:
    ok, reason = looks_ok(GOOD_CODE, TPCH_TABLES)
    assert ok, f"expected accept, got {reason}"


def test_looks_ok_accepts_code_with_alias_column() -> None:
    """Columns created via .alias() during the chain should be considered valid."""
    code = (
        'result = (orders.group_by("o_custkey")'
        '.agg(pl.len().alias("n_orders"))'
        '.sort("n_orders", descending=True))'
    )
    ok, _ = looks_ok(code, TPCH_TABLES)
    assert ok


def test_looks_ok_accepts_multi_table_join() -> None:
    code = (
        'result = customer.join(orders, left_on="c_custkey", right_on="o_custkey")'
        '.select(["c_name", "o_totalprice"])'
    )
    ok, _ = looks_ok(code, TPCH_TABLES)
    assert ok


# ---------------------------------------------------------------------------
# looks_ok — edge cases
# ---------------------------------------------------------------------------

def test_looks_ok_rejects_empty() -> None:
    ok, reason = looks_ok("", TPCH_TABLES)
    assert not ok and reason == "empty"


def test_looks_ok_rejects_whitespace_only() -> None:
    ok, reason = looks_ok("   \n\t  ", TPCH_TABLES)
    assert not ok and reason == "empty"


def test_looks_ok_rejects_syntax_error() -> None:
    ok, reason = looks_ok("result = customer.filter(", TPCH_TABLES)
    assert not ok and reason.startswith("syntax_error")


def test_looks_ok_rejects_missing_result_assignment() -> None:
    code = 'customer.filter(pl.col("c_custkey") > 10)'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and reason == "missing_result_assignment"


def test_looks_ok_rejects_hallucinated_with_column() -> None:
    code = 'result = customer.with_column(pl.col("c_custkey") + 1)'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and "with_column" in reason


def test_looks_ok_rejects_pl_desc() -> None:
    code = 'result = customer.sort(pl.desc("c_custkey"))'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and "pl.desc" in reason


def test_looks_ok_rejects_df_dot_len_attribute() -> None:
    code = 'result = customer.len'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and ".len" in reason


def test_looks_ok_rejects_groupby_pandas_style() -> None:
    code = 'result = customer.groupby("c_nationkey").agg(pl.len())'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and "groupby" in reason


def test_looks_ok_rejects_unknown_column() -> None:
    code = 'result = customer.filter(pl.col("not_a_real_column") > 0)'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and "unknown_column" in reason and "not_a_real_column" in reason


def test_looks_ok_ignores_column_check_when_schema_empty() -> None:
    """If tables dict is empty (no schema available), we can't validate column
    refs, so we shouldn't reject on that basis."""
    code = 'result = df.filter(pl.col("anything") > 0)'
    ok, _ = looks_ok(code, {})
    assert ok


def test_looks_ok_accepts_str_contains() -> None:
    """`.str.contains(` is the modern form and must NOT be flagged."""
    code = 'result = customer.filter(pl.col("c_name").str.contains("foo"))'
    ok, _ = looks_ok(code, TPCH_TABLES)
    assert ok, "str.contains is valid modern Polars"


def test_looks_ok_rejects_bare_contains_on_expr() -> None:
    code = 'result = customer.filter(pl.col("c_name").contains("foo"))'
    ok, reason = looks_ok(code, TPCH_TABLES)
    assert not ok and ".contains" in reason


# ---------------------------------------------------------------------------
# detect_hallucinations
# ---------------------------------------------------------------------------

def test_detect_hallucinations_clean_code_returns_empty() -> None:
    assert detect_hallucinations(GOOD_CODE) == []


def test_detect_hallucinations_empty_input() -> None:
    assert detect_hallucinations("") == []


def test_detect_hallucinations_finds_multiple() -> None:
    code = 'result = df.with_column(pl.desc("x")).len'
    issues = detect_hallucinations(code)
    joined = " ".join(issues)
    assert "with_column" in joined
    assert "pl.desc" in joined
    assert ".len" in joined


def test_detect_hallucinations_preserves_order() -> None:
    """Labels should come back as human-readable messages usable in feedback."""
    issues = detect_hallucinations('df.with_column(x)')
    assert len(issues) == 1
    assert "with_columns" in issues[0]  # should suggest the fix


# ---------------------------------------------------------------------------
# _collect_valid_columns — both schema formats
# ---------------------------------------------------------------------------

def test_collect_columns_enriched_format() -> None:
    cols = _collect_valid_columns(TPCH_TABLES)
    assert cols == {
        "c_custkey", "c_name", "c_nationkey",
        "o_orderkey", "o_custkey", "o_totalprice",
    }


def test_collect_columns_flat_legacy_format() -> None:
    """Some older code passes {"table": {col: type}} without the "columns" key."""
    tables = {"t": {"x": "Int64", "y": "Utf8"}}
    cols = _collect_valid_columns(tables)
    assert cols == {"x", "y"}


def test_collect_columns_list_format() -> None:
    """Very old format with just a list of column names."""
    tables = {"t": ["x", "y", "z"]}
    cols = _collect_valid_columns(tables)
    assert cols == {"x", "y", "z"}


def test_collect_columns_empty_tables() -> None:
    assert _collect_valid_columns({}) == set()


# ---------------------------------------------------------------------------
# run_cascade — orchestrator behavior
# ---------------------------------------------------------------------------

def test_cascade_returns_fast_when_l1_ok() -> None:
    model = MockModel(fast=GOOD_CODE)
    res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    assert res.level == "fast"
    assert res.code == GOOD_CODE
    assert model.calls == ["fast"]  # L2 & L3 not called


def test_cascade_falls_through_to_constrained_on_l1_hallucination() -> None:
    """L1 produces hallucinated code → L2 should be tried."""
    bad = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad, constrained=GOOD_CODE)
    # Force-enable L2 even though CONSTRAINED_AVAILABLE is False in this env:
    # use monkeypatching by passing a model that stubs generate_constrained.
    # But run_cascade checks CONSTRAINED_AVAILABLE module-level. We patch it.
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = True
    try:
        res = run_cascade(model, "q", TPCH_TABLES)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    assert res.level == "constrained"
    assert res.code == GOOD_CODE
    assert model.calls == ["fast", "constrained"]


def test_cascade_falls_through_to_retry_when_l2_returns_bad_code() -> None:
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    bad2 = 'result = customer.groupby("c_nationkey").agg(pl.len())'
    model = MockModel(fast=bad1, constrained=bad2, retry=GOOD_CODE)
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = True
    try:
        res = run_cascade(model, "q", TPCH_TABLES)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    assert res.level == "retry"
    assert res.code == GOOD_CODE
    assert model.calls == ["fast", "constrained", "retry"]


def test_cascade_falls_back_to_safe_default_when_all_levels_fail_validation() -> None:
    """When L1/L2/L3 all fail looks_ok, the fallback must return a SAFE
    DEFAULT (not one of the broken candidates). Returning broken code like
    `)` or a tuple-suffix ruins polars.bench scoring — better to return
    `result = <first_table>` which at least parses, runs, and returns a DF."""
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    bad2 = 'result = customer.groupby("x")'
    bad3 = 'result = customer.len'
    model = MockModel(fast=bad1, constrained=bad2, retry=bad3)
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = True
    try:
        res = run_cascade(model, "q", TPCH_TABLES)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    # None of the bad candidates should be returned. Safe default wins.
    assert res.level == "fallback_default"
    assert res.code.startswith("result = ")
    assert res.code not in (bad1, bad2, bad3)


def test_cascade_fallback_returns_candidate_that_passes_looks_ok() -> None:
    """If a candidate secretly passes looks_ok (e.g. retry produces good code
    but something else in the cascade logic rejected it), the fallback should
    pick that candidate, not the safe default."""
    # Craft: L1 fails looks_ok (hallucination), L2 disabled, L3 produces GOOD
    # code. Normally run_cascade would return L3 at the "retry" level, not
    # the fallback. But if it somehow reached fallback, it should pick L3.
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad1, retry=GOOD_CODE)
    res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    # Should actually return at "retry" level (not fallback) — but either way
    # the code must be GOOD_CODE.
    assert res.code == GOOD_CODE
    assert res.level in ("retry", "fallback_ok")


def test_cascade_skips_l2_when_constrained_unavailable() -> None:
    """Case Jacques: llguidance not installed → L2 auto-skipped, go direct to L3."""
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad1, retry=GOOD_CODE)
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = False
    try:
        res = run_cascade(model, "q", TPCH_TABLES)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    assert res.level == "retry"
    assert "constrained" not in model.calls, "L2 should not have been called"
    assert model.calls == ["fast", "retry"]


def test_cascade_skips_l2_when_disabled_via_flag() -> None:
    """disable_constrained=True forces skip even if backend is available."""
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad1, retry=GOOD_CODE)
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = True
    try:
        res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    assert res.level == "retry"
    assert "constrained" not in model.calls


def test_cascade_handles_l2_exception_gracefully() -> None:
    """If generate_constrained raises (dep issue, OOM…), we fall through to L3."""
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(
        fast=bad1,
        retry=GOOD_CODE,
        constrained_raises=RuntimeError("llguidance glue broken"),
    )
    import gemma_cascade
    prev = gemma_cascade.CONSTRAINED_AVAILABLE
    gemma_cascade.CONSTRAINED_AVAILABLE = True
    try:
        res = run_cascade(model, "q", TPCH_TABLES)
    finally:
        gemma_cascade.CONSTRAINED_AVAILABLE = prev
    assert res.level == "retry"
    assert res.code == GOOD_CODE


def test_cascade_handles_l3_exception_gracefully() -> None:
    """If even retry crashes, fallback must still return something non-empty
    AND statically valid (not a broken L1 code we just rejected)."""
    bad1 = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad1, retry_raises=RuntimeError("cuda OOM"))
    res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    assert res.level == "fallback_default"
    assert res.code.startswith("result = ")  # safe default, NOT bad1


def test_cascade_fallback_uses_safe_default_when_all_empty() -> None:
    """Even if the model returns empty strings, fallback gives a runnable stub."""
    model = MockModel(fast="", retry="")
    res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    assert res.level == "fallback_default"
    assert res.code.startswith("result = ")


def test_cascade_records_l1_debug_info() -> None:
    """CascadeResult must carry the l1_code and l1_reason for debugging."""
    bad = 'result = customer.with_column(pl.col("c_custkey"))'
    model = MockModel(fast=bad, retry=GOOD_CODE)
    res = run_cascade(model, "q", TPCH_TABLES, disable_constrained=True)
    assert res.l1_code == bad
    assert "with_column" in res.l1_reason


def test_cascade_result_is_dataclass() -> None:
    res = CascadeResult(code="x", level="fast", reason="ok")
    assert res.code == "x"
    assert res.level == "fast"


# ---------------------------------------------------------------------------
# try_mock_execute — mock-execution catches common runtime errors
# ---------------------------------------------------------------------------

def test_mock_exec_accepts_valid_code() -> None:
    code = 'result = customer.filter(pl.col("c_custkey") > 10).select("c_name")'
    ok, err = try_mock_execute(code, TPCH_TABLES)
    assert ok, f"expected ok, got: {err}"


def test_mock_exec_rejects_unknown_column() -> None:
    code = 'result = customer.filter(pl.col("fake_col") > 0)'
    ok, err = try_mock_execute(code, TPCH_TABLES)
    assert not ok
    assert "ColumnNotFound" in err or "fake_col" in err


def test_mock_exec_rejects_missing_result() -> None:
    code = 'x = customer.filter(pl.col("c_custkey") > 0)'
    ok, err = try_mock_execute(code, TPCH_TABLES)
    assert not ok
    assert "result" in err.lower()


def test_mock_exec_rejects_empty_code() -> None:
    ok, err = try_mock_execute("", TPCH_TABLES)
    assert not ok


def test_mock_exec_rejects_syntax_error() -> None:
    ok, err = try_mock_execute("result = customer.filter(", TPCH_TABLES)
    assert not ok
    assert "Syntax" in err or "SyntaxError" in err


def test_mock_exec_catches_join_duplicate_error() -> None:
    """Joining on columns that produce a name collision should be caught."""
    # customer has c_name, nation (not in our TPCH_TABLES) would have n_name.
    # Instead use a self-inducing case: two tables with same non-key column.
    tables = {
        "a": {"columns": {"id": "Int64", "name": "Utf8"}, "n_rows": 10},
        "b": {"columns": {"id": "Int64", "name": "Utf8"}, "n_rows": 10},
    }
    # Join that may create a "name_right" suffix is OK, but refer to a
    # dropped column after the join to exhibit a failure
    code = 'result = a.join(b, on="id").filter(pl.col("missing") > 0)'
    ok, err = try_mock_execute(code, tables)
    assert not ok
    assert "ColumnNotFound" in err or "missing" in err


def test_mock_exec_handles_list_format_schema() -> None:
    """Fallback: schema as bare list should not crash."""
    tables = {"t": ["x", "y"]}
    # Valid column ref
    code = 'result = t.select("x")'
    ok, err = try_mock_execute(code, tables)
    assert ok


def test_mock_exec_unknown_dtype_falls_back_to_utf8() -> None:
    """An unfamiliar type string should not crash mock setup."""
    tables = {"t": {"columns": {"x": "NotARealDtype"}, "n_rows": 1}}
    code = 'result = t.select("x")'
    ok, _ = try_mock_execute(code, tables)
    assert ok


# ---------------------------------------------------------------------------
# run_cascade_with_exec_retry
# ---------------------------------------------------------------------------

def test_exec_retry_returns_cascade_result_when_mock_exec_passes() -> None:
    """If the cascade's output already passes mock exec, no retry needed."""
    model = MockModel(fast=GOOD_CODE)
    res = run_cascade_with_exec_retry(
        model, "q", TPCH_TABLES, disable_constrained=True, max_retries=2
    )
    assert res.level == "fast"
    assert model.calls == ["fast"]  # no retry triggered


def test_exec_retry_triggers_when_cascade_code_fails_mock_exec() -> None:
    """Code that passes looks_ok but fails mock exec must trigger the exec-retry.

    `.nonexistent_method()` is not caught by our static hallucination regex,
    so looks_ok passes at L1, but mock exec raises AttributeError.
    """
    static_ok_but_bad = 'result = customer.nonexistent_method()'
    model = MockModel(fast=static_ok_but_bad, retry=GOOD_CODE)
    res = run_cascade_with_exec_retry(
        model, "q", TPCH_TABLES, disable_constrained=True, max_retries=2
    )
    assert res.level.startswith("exec_retry")
    assert res.code == GOOD_CODE
    assert model.calls.count("retry") >= 1


def test_exec_retry_respects_max_retries() -> None:
    """After N retries, give up and return the last code (even if bad)."""
    static_ok_but_bad = 'result = customer.nonexistent_method()'
    static_ok_but_bad2 = 'result = customer.another_nonexistent()'
    model = MockModel(fast=static_ok_but_bad, retry=static_ok_but_bad2)
    res = run_cascade_with_exec_retry(
        model, "q", TPCH_TABLES, disable_constrained=True, max_retries=2
    )
    # retry called exactly max_retries times, returns the last bad code
    assert model.calls.count("retry") == 2


def test_exec_retry_handles_model_exception_gracefully() -> None:
    """If the retry call itself raises, stop gracefully and return the prior best."""
    static_ok_but_bad = 'result = customer.nonexistent_method()'
    model = MockModel(fast=static_ok_but_bad, retry_raises=RuntimeError("oom"))
    res = run_cascade_with_exec_retry(
        model, "q", TPCH_TABLES, disable_constrained=True, max_retries=2
    )
    # should not crash; returns the bad code
    assert res.code == static_ok_but_bad


# ---------------------------------------------------------------------------
# build_structured_feedback (E) — reformulate Polars errors for the LLM
# ---------------------------------------------------------------------------

def test_feedback_extracts_missing_col_and_valid_list() -> None:
    """The sweet spot: ColumnNotFoundError with both the missing name and
    the valid list — must produce a clean, actionable feedback."""
    err = (
        'ColumnNotFoundError: unable to find column "user_id"; '
        'valid columns: ["customer_id", "first_name", "last_name"]'
    )
    fb = build_structured_feedback(err)
    assert '"user_id"' in fb
    assert "customer_id" in fb and "first_name" in fb
    assert "EXCLUSIVELY" in fb or "exclusively" in fb.lower()


def test_feedback_handles_column_not_found_without_valid_list() -> None:
    """Sometimes Polars may not include the valid-columns list. Fallback gracefully."""
    err = "ColumnNotFoundError: something about a column"
    fb = build_structured_feedback(err)
    assert "column" in fb.lower()
    assert err[:50] in fb  # raw error is quoted


def test_feedback_handles_duplicate_error_join() -> None:
    err = "DuplicateError: column with name 'name_right' already exists"
    fb = build_structured_feedback(err)
    assert "DuplicateError" in fb or "collision" in fb.lower()
    assert "suffix" in fb  # the fix hint


def test_feedback_handles_schema_error() -> None:
    err = "SchemaError: could not evaluate '<=' comparison between List and Int"
    fb = build_structured_feedback(err)
    assert "SchemaError" in fb or "type" in fb.lower()


def test_feedback_handles_attribute_error() -> None:
    err = "AttributeError: 'DataFrame' object has no attribute 'len'"
    fb = build_structured_feedback(err)
    assert "attribute" in fb.lower() or "API" in fb


def test_feedback_empty_error_is_safe() -> None:
    fb = build_structured_feedback("")
    assert "unknown" in fb.lower() or "rewrite" in fb.lower()


def test_feedback_unknown_error_falls_back_to_raw() -> None:
    err = "TotallyMadeUpError: something wrong"
    fb = build_structured_feedback(err)
    assert "something wrong" in fb


def test_feedback_truncates_very_long_raw_errors() -> None:
    """Don't blow up the context window with a giant error."""
    err = "SomeError: " + "X" * 10000
    fb = build_structured_feedback(err)
    assert len(fb) < 600  # capped


# ---------------------------------------------------------------------------
# exec-retry uses structured feedback end-to-end
# ---------------------------------------------------------------------------

class _CaptureModel(MockModel):
    """Captures the feedback string passed to generate_with_feedback."""
    def __init__(self, fast: str, retry: str) -> None:
        super().__init__(fast=fast, retry=retry)
        self.last_feedback: str | None = None
        self.last_previous_code: str | None = None

    def generate_with_feedback(self, message, tables, previous_code, feedback):
        self.last_previous_code = previous_code
        self.last_feedback = feedback
        return super().generate_with_feedback(message, tables, previous_code, feedback)


def test_exec_retry_passes_structured_feedback_not_raw() -> None:
    """The EXEC retry (not L3) should receive the structured feedback built
    from the Polars error. To exercise exec-retry specifically, the L1 code
    must pass looks_ok but fail mock_execute — we use .nonexistent_method()
    which slips past static checks but raises AttributeError at exec time."""
    static_ok_but_bad = 'result = customer.nonexistent_method()'
    model = _CaptureModel(fast=static_ok_but_bad, retry=GOOD_CODE)
    run_cascade_with_exec_retry(
        model, "q", TPCH_TABLES, disable_constrained=True, max_retries=2
    )
    assert model.last_feedback is not None
    # For AttributeError, the structured feedback should reference the API rules
    assert "attribute" in model.last_feedback.lower() or "API" in model.last_feedback
    # And must contain the raw error info so the model can pinpoint the issue
    assert "nonexistent_method" in model.last_feedback


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests() -> list:
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    tests = _all_tests()
    passed, failed = 0, 0
    failures: list[tuple[str, str]] = []
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  ok     {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL   {name}: {e}")
            failed += 1
            failures.append((name, str(e)))
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR  {name}: {type(e).__name__}: {e}")
            failed += 1
            failures.append((name, f"{type(e).__name__}: {e}"))

    print()
    print(f"  {passed} passed, {failed} failed ({len(tests)} total)")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
