"""Tests for the Gemma 4 E2B benchmark plumbing.

Covers what can be validated WITHOUT loading the model (no GPU required):
  - gemma_prompt.py : SYSTEM_PROMPT, FEWSHOT, format_schema, format_user_turn
  - benchmark_gemma.py : AST-level structure (class methods, CLI args)
  - build_grammar compatibility with the prompt_schemas format we emit

The actual model loading + generation is tested on the VM by running the
benchmark with `--oracle --limit 3` and `--limit 3`. This file catches plumbing
regressions locally without needing the 10GB Gemma checkpoint.

Run:
    python -m tests.test_gemma_prompt           # all tests
    python -m pytest tests/test_gemma_prompt.py # via pytest if installed
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gemma_prompt import (  # noqa: E402
    FEWSHOT,
    SYSTEM_PROMPT,
    _format_strict_block,
    format_schema,
    format_user_turn,
)


# ---------------------------------------------------------------------------
# gemma_prompt.py — pure Python
# ---------------------------------------------------------------------------

def test_system_prompt_non_empty() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 200, "system prompt suspiciously short"


def test_system_prompt_contains_modern_api_rules() -> None:
    """The cheat sheet must list the key anti-patterns we observed in baseline."""
    for must_mention in [
        "with_columns",
        "pl.desc",
        "group_by",
        "str.contains",
        "rank(method",
    ]:
        assert must_mention in SYSTEM_PROMPT, f"missing from cheat sheet: {must_mention}"


def test_fewshot_structure() -> None:
    assert len(FEWSHOT) >= 3, "expected at least 3 few-shot examples"
    for i, (tables, question, code) in enumerate(FEWSHOT):
        assert isinstance(tables, dict) and tables, f"fewshot[{i}] tables empty"
        assert isinstance(question, str) and question.strip(), f"fewshot[{i}] no question"
        assert isinstance(code, str) and code.strip(), f"fewshot[{i}] no code"
        assert "result =" in code or "result=" in code, (
            f"fewshot[{i}] missing `result =` assignment"
        )


def test_fewshot_code_is_valid_python() -> None:
    for i, (_t, _q, code) in enumerate(FEWSHOT):
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise AssertionError(f"fewshot[{i}] code does not parse: {e}\n{code}")


def test_fewshot_uses_modern_polars_only() -> None:
    """Few-shot code must not contain any of the hallucinated APIs we teach against."""
    forbidden = [
        (".with_column(", "should be .with_columns("),
        ("pl.desc(", "pl.desc does not exist; use sort(descending=True)"),
        ("groupby(", "should be group_by("),
    ]
    for i, (_t, _q, code) in enumerate(FEWSHOT):
        for pat, reason in forbidden:
            assert pat not in code, f"fewshot[{i}] contains {pat!r}: {reason}"


def test_fewshot_covers_window_pattern() -> None:
    """At least one few-shot should demonstrate the rank/over pattern
    (window functions were the weakest tag in the Gemma baseline)."""
    has_window = any(
        ".rank(" in code and ".over(" in code
        for _t, _q, code in FEWSHOT
    )
    assert has_window, "no few-shot covers window/rank pattern"


# ---------------------------------------------------------------------------
# format_schema — A3 enriched schema format
# ---------------------------------------------------------------------------

def test_format_schema_with_n_rows() -> None:
    tables = {
        "customer": {"columns": {"c_custkey": "Int64", "c_name": "Utf8"}, "n_rows": 1500},
    }
    rendered = format_schema(tables)
    assert "customer" in rendered
    assert "1500 rows" in rendered
    assert "c_custkey: Int64" in rendered
    assert "c_name: Utf8" in rendered


def test_format_schema_multiple_tables() -> None:
    tables = {
        "customer": {"columns": {"c_custkey": "Int64"}, "n_rows": 100},
        "orders": {"columns": {"o_orderkey": "Int64"}, "n_rows": 1000},
    }
    rendered = format_schema(tables)
    assert "customer" in rendered and "orders" in rendered
    assert "100 rows" in rendered and "1000 rows" in rendered


def test_format_schema_without_n_rows_defensive() -> None:
    """A legacy record without n_rows must not crash."""
    tables = {"customer": {"columns": {"c_custkey": "Int64"}}}
    rendered = format_schema(tables)
    assert "customer" in rendered
    assert "rows" not in rendered, "should not invent an n_rows annotation"


def test_format_schema_flat_legacy_format() -> None:
    """Very old format where meta is directly the columns dict."""
    tables = {"customer": {"c_custkey": "Int64", "c_name": "Utf8"}}
    rendered = format_schema(tables)
    assert "c_custkey: Int64" in rendered
    assert "c_name: Utf8" in rendered


def test_format_schema_empty_columns() -> None:
    """Edge case: table declared but no columns. Should render the header and no column lines."""
    tables = {"empty": {"columns": {}, "n_rows": 0}}
    rendered = format_schema(tables)
    assert "empty" in rendered
    assert "0 rows" in rendered


def test_format_user_turn_wraps_schema_and_question() -> None:
    tables = {"t": {"columns": {"x": "Int64"}, "n_rows": 10}}
    q = "What is x?"
    turn = format_user_turn(tables, q)
    assert q in turn
    assert "Datasets:" in turn
    assert "- t (10 rows)" in turn


def test_format_user_turn_preserves_special_chars_in_question() -> None:
    """Questions with quotes, newlines, emojis must not break the output."""
    tables = {"t": {"columns": {"x": "Int64"}, "n_rows": 10}}
    q = 'Filter where name == "O\'Brien" and count\nrows 😀'
    turn = format_user_turn(tables, q)
    assert q in turn


# ---------------------------------------------------------------------------
# Strict block (D) — stronger schema emphasis on the real user turn
# ---------------------------------------------------------------------------

def test_format_user_turn_default_has_no_strict_block() -> None:
    """Few-shot turns should stay compact — no STRICT by default."""
    tables = {"t": {"columns": {"x": "Int64"}, "n_rows": 10}}
    turn = format_user_turn(tables, "Q?")
    assert "STRICT" not in turn


def test_format_user_turn_strict_appends_column_reminder() -> None:
    tables = {
        "customer": {
            "columns": {"customer_id": "Int64", "first_name": "Utf8"},
            "n_rows": 100,
        },
        "payment": {
            "columns": {"payment_id": "Int64", "amount": "Float64"},
            "n_rows": 50,
        },
    }
    turn = format_user_turn(tables, "List top customers", strict=True)
    # Strict block uses passive wording (describes what EXISTS, not what to USE)
    # to avoid the "select all these columns" misreading we observed with
    # prescriptive wording.
    assert "Available columns" in turn
    assert "ColumnNotFoundError" in turn  # the failure mode is named explicitly
    assert "customer_id" in turn and "first_name" in turn
    assert "payment_id" in turn and "amount" in turn
    # Per-table grouping preserved so the model can associate cols with tables
    assert "customer:" in turn
    assert "payment:" in turn
    # And an explicit reminder that selecting all is wrong
    assert "not all" in turn.lower()


def test_format_strict_block_empty_tables_returns_empty() -> None:
    assert _format_strict_block({}) == ""


def test_format_strict_block_empty_columns_in_table_is_skipped() -> None:
    """A table with an empty columns dict should be silently skipped, not crash."""
    tables = {"empty": {"columns": {}, "n_rows": 0}}
    assert _format_strict_block(tables) == ""


def test_format_strict_block_handles_list_format() -> None:
    """Legacy list-style schema must still produce a reminder."""
    tables = {"t": ["x", "y", "z"]}
    block = _format_strict_block(tables)
    assert "t:" in block
    assert "x" in block and "y" in block and "z" in block


def test_format_user_turn_strict_false_matches_no_strict() -> None:
    """strict=False explicitly should behave like the default."""
    tables = {"t": {"columns": {"x": "Int64"}, "n_rows": 1}}
    assert format_user_turn(tables, "Q", strict=False) == format_user_turn(tables, "Q")


def test_strict_block_appears_after_question_not_before() -> None:
    """Recency bias — the available-columns reminder must be AFTER the
    question so it's among the most recent tokens the model sees."""
    tables = {"t": {"columns": {"x": "Int64"}, "n_rows": 1}}
    turn = format_user_turn(tables, "THE_QUESTION", strict=True)
    idx_q = turn.index("THE_QUESTION")
    idx_s = turn.index("Available columns")
    assert idx_q < idx_s, "reminder block must come after the question"


# ---------------------------------------------------------------------------
# benchmark_gemma.py — AST-level structure
# ---------------------------------------------------------------------------

def _parse_file(name: str) -> ast.Module:
    src = (REPO_ROOT / name).read_text()
    return ast.parse(src)


def test_benchmark_gemma_parses() -> None:
    _parse_file("benchmark_gemma.py")  # raises if syntax error


def test_gemma_model_parses() -> None:
    _parse_file("gemma_model.py")


def test_gemma_model_class_has_required_methods() -> None:
    """GemmaModel is now in gemma_model.py (extracted from benchmark_gemma.py)."""
    tree = _parse_file("gemma_model.py")
    methods: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GemmaModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods[item.name] = [a.arg for a in item.args.args]
            break
    else:
        raise AssertionError("class GemmaModel not found in gemma_model.py")

    for required in (
        "__init__",
        "_build_prompt",
        "_ensure_outlines",
        "generate",
        "generate_constrained",
        "generate_with_feedback",
    ):
        assert required in methods, f"GemmaModel missing method: {required}"

    # generate_constrained must accept the same (message, tables) contract as generate
    assert "message" in methods["generate_constrained"]
    assert "tables" in methods["generate_constrained"]

    # generate_with_feedback must accept previous_code + feedback in addition to message/tables
    fb_args = methods["generate_with_feedback"]
    for required_arg in ("message", "tables", "previous_code", "feedback"):
        assert required_arg in fb_args, f"generate_with_feedback missing arg: {required_arg}"


def test_cli_has_constrained_flag() -> None:
    """Ensure main() registered --constrained via add_argument."""
    tree = _parse_file("benchmark_gemma.py")
    flags_seen: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags_seen.add(arg.value)
    for expected in ("--oracle", "--constrained", "--limit", "--out"):
        assert expected in flags_seen, f"CLI flag missing: {expected}"


def test_benchmark_gemma_imports_gemma_model() -> None:
    """After refactor, benchmark_gemma.py imports GemmaModel from gemma_model."""
    tree = _parse_file("benchmark_gemma.py")
    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.setdefault(node.module, set()).update(n.name for n in node.names)

    assert "gemma_model" in imports, "benchmark_gemma.py should import from gemma_model"
    assert "GemmaModel" in imports["gemma_model"]


def test_gemma_model_imports_prompt_and_grammar() -> None:
    """gemma_model.py is the central consumer of gemma_prompt + polars_grammar."""
    tree = _parse_file("gemma_model.py")
    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.setdefault(node.module, set()).update(n.name for n in node.names)

    assert "gemma_prompt" in imports, "should import from gemma_prompt"
    for sym in ("FEWSHOT", "SYSTEM_PROMPT", "format_user_turn"):
        assert sym in imports["gemma_prompt"], f"missing symbol {sym}"
    assert "dataset.polars_grammar" in imports, "should import build_grammar"
    assert "build_grammar" in imports["dataset.polars_grammar"]


# ---------------------------------------------------------------------------
# Integration: build_grammar + our prompt_schemas shape
# ---------------------------------------------------------------------------

def _simulated_prompt_schemas(rec_tables: dict) -> dict:
    """Mirror exactly what run() and evaluate_one() build."""
    return {
        name: {"columns": schema["columns"], "n_rows": schema.get("n_rows")}
        for name, schema in rec_tables.items()
    }


def test_simulated_prompt_schemas_shape() -> None:
    rec_tables = {
        "customer": {"columns": {"c_custkey": "Int64"}, "n_rows": 1500},
        "orders": {"columns": {"o_orderkey": "Int64"}, "n_rows": 15000},
    }
    schemas = _simulated_prompt_schemas(rec_tables)
    assert schemas["customer"]["columns"] == {"c_custkey": "Int64"}
    assert schemas["customer"]["n_rows"] == 1500
    # format_schema must handle it
    rendered = format_schema(schemas)
    assert "1500 rows" in rendered


def test_build_grammar_accepts_enriched_schema() -> None:
    try:
        from dataset.polars_grammar import build_grammar
    except ImportError as e:
        raise AssertionError(f"cannot import build_grammar: {e}")

    schemas = _simulated_prompt_schemas({
        "customer": {"columns": {"c_custkey": "Int64", "c_name": "Utf8"}, "n_rows": 1500},
        "orders": {"columns": {"o_orderkey": "Int64", "o_custkey": "Int64"}, "n_rows": 15000},
    })
    grammar = build_grammar(schemas)
    assert isinstance(grammar, str) and len(grammar) > 500
    assert "customer" in grammar and "orders" in grammar
    assert "c_custkey" in grammar and "o_orderkey" in grammar
    assert "with_columns" in grammar, "modern API must be in grammar"


def test_build_grammar_rejects_empty_tables() -> None:
    from dataset.polars_grammar import build_grammar
    try:
        build_grammar({})
    except ValueError:
        return
    raise AssertionError("build_grammar should reject empty tables dict")


# ---------------------------------------------------------------------------
# End-to-end format check on real seeds if available
# ---------------------------------------------------------------------------

def test_format_schema_on_real_seeds() -> None:
    """Run format_schema on every record in data/seeds.jsonl to catch any edge
    case (missing key, weird type) before hitting the model on the VM."""
    seeds_path = REPO_ROOT / "data" / "seeds.jsonl"
    if not seeds_path.exists():
        # Skip silently if seeds haven't been generated locally (the VM has them)
        return

    n_records = 0
    for line in seeds_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        schemas = _simulated_prompt_schemas(rec["tables"])
        rendered = format_user_turn(schemas, rec["question"])
        assert "Datasets:" in rendered
        assert rec["question"] in rendered
        n_records += 1
    assert n_records > 0, "no seeds found to test against"


# ---------------------------------------------------------------------------
# Runner — works without pytest
# ---------------------------------------------------------------------------

def _all_tests() -> list:
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    tests = _all_tests()
    passed, failed, skipped = 0, 0, 0
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
            msg = f"{type(e).__name__}: {e}"
            # import errors on optional deps → skip rather than fail
            if isinstance(e, ImportError):
                print(f"  skip   {name} ({msg})")
                skipped += 1
            else:
                print(f"  ERROR  {name}: {msg}")
                failed += 1
                failures.append((name, msg))

    print()
    print(f"  {passed} passed, {failed} failed, {skipped} skipped ({len(tests)} total)")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
