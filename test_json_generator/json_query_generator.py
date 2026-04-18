"""JSON-based query generation with Gemma 4 E2B.

Instead of generating raw Polars code, the model produces a structured JSON
query plan constrained by a Pydantic schema via Outlines. A separate
json_to_polars() function converts the plan to executable Polars code.

Flow:
    question + schema → GemmaModel (JSON-constrained) → QueryPlan (JSON)
                                                       → json_to_polars()
                                                       → Polars code string
"""
from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Annotated, Any, Literal, Union
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Query plan schema
# ---------------------------------------------------------------------------

class CompareOp(str, Enum):
    eq = "=="
    ne = "!="
    lt = "<"
    le = "<="
    gt = ">"
    ge = ">="


class Condition(BaseModel):
    column: str
    op: CompareOp
    value: Union[str, int, float, bool]


class FilterOp(BaseModel):
    type: Literal["filter"]
    conditions: list[Condition]
    combine: Literal["and", "or"] = "and"


class Aggregation(BaseModel):
    agg: Literal["sum", "mean", "min", "max", "len", "n_unique", "first", "last"]
    column: str  # "*" means pl.len() (row count)
    alias: str


class GroupByOp(BaseModel):
    type: Literal["group_by"]
    columns: list[str]
    aggregations: list[Aggregation]


class SortColumn(BaseModel):
    name: str
    descending: bool = False


class SortOp(BaseModel):
    type: Literal["sort"]
    columns: list[SortColumn]


class SelectOp(BaseModel):
    type: Literal["select"]
    columns: list[str]


class JoinOp(BaseModel):
    type: Literal["join"]
    table: str
    on: str = ""          # shared key name; empty string means use left_on/right_on
    left_on: str = ""
    right_on: str = ""
    how: Literal["inner", "left", "outer"] = "inner"


class WithColumnExpr(BaseModel):
    column: str
    expr: Literal["rank"]
    method: Literal["dense", "ordinal", "average", "min", "max"] = "dense"
    descending: bool = True
    over: str = ""    # empty string means no window partition
    alias: str


class WithColumnsOp(BaseModel):
    type: Literal["with_columns"]
    expressions: list[WithColumnExpr]


class HeadOp(BaseModel):
    type: Literal["head"]
    n: int


# Plain Union — no discriminator field, so Pydantic emits a flat anyOf schema
# that llguidance handles correctly (discriminator+oneOf breaks constrained gen).
Operation = Union[FilterOp, GroupByOp, SortOp, SelectOp, JoinOp, WithColumnsOp, HeadOp]


class QueryPlan(BaseModel):
    source_table: str
    operations: list[Operation]


# ---------------------------------------------------------------------------
# JSON → Polars converter
# ---------------------------------------------------------------------------

def _render_value(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def _render_condition(c: Condition) -> str:
    return f'(pl.col("{c.column}") {c.op.value} {_render_value(c.value)})'


def _render_filter(op: FilterOp) -> str:
    joiner = " & " if op.combine == "and" else " | "
    expr = joiner.join(_render_condition(c) for c in op.conditions)
    return f".filter({expr})"


def _render_agg(a: Aggregation) -> str:
    if a.agg == "len" or a.column == "*":
        base = "pl.len()"
    else:
        base = f'pl.col("{a.column}").{a.agg}()'
    return f'{base}.alias("{a.alias}")'


def _render_group_by(op: GroupByOp) -> str:
    cols = ", ".join(f'"{c}"' for c in op.columns)
    aggs = ", ".join(_render_agg(a) for a in op.aggregations)
    return f".group_by({cols})\n    .agg({aggs})"


def _render_sort(op: SortOp) -> str:
    if len(op.columns) == 1:
        sc = op.columns[0]
        return f'.sort("{sc.name}", descending={str(sc.descending)})'
    by = "[" + ", ".join(f'"{sc.name}"' for sc in op.columns) + "]"
    desc = "[" + ", ".join(str(sc.descending) for sc in op.columns) + "]"
    return f".sort({by}, descending={desc})"


def _render_select(op: SelectOp) -> str:
    cols = ", ".join(f'"{c}"' for c in op.columns)
    return f".select({cols})"


def _render_join(op: JoinOp) -> str:
    if op.on:
        key = f'on="{op.on}"'
    else:
        key = f'left_on="{op.left_on}", right_on="{op.right_on}"'
    return f'.join({op.table}, {key}, how="{op.how}")'


def _render_with_columns(op: WithColumnsOp) -> str:
    parts = []
    for e in op.expressions:
        base = f'pl.col("{e.column}").rank(method="{e.method}", descending={str(e.descending)})'
        if e.over:  # empty string is falsy → no .over()
            base += f'.over("{e.over}")'
        base += f'.alias("{e.alias}")'
        parts.append(base)
    return ".with_columns(\n        " + ",\n        ".join(parts) + "\n    )"


def _render_head(op: HeadOp) -> str:
    return f".head({op.n})"


_RENDERERS = {
    "filter": _render_filter,
    "group_by": _render_group_by,
    "sort": _render_sort,
    "select": _render_select,
    "join": _render_join,
    "with_columns": _render_with_columns,
    "head": _render_head,
}


def json_to_polars(plan: QueryPlan | dict) -> str:
    """Convert a QueryPlan (or its dict representation) to a Polars code string."""
    if isinstance(plan, dict):
        plan = QueryPlan.model_validate(plan)

    lines = [f"result = (", f"    {plan.source_table}"]
    for op in plan.operations:
        renderer = _RENDERERS[op.type]
        rendered = renderer(op)  # type: ignore[arg-type]
        for line in rendered.splitlines():
            lines.append(f"    {line}")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemma-based JSON generator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a data query planner. Output only a JSON object — no prose, no markdown fences.

CRITICAL: "source_table" must be the bare table name exactly as it appears in "Datasets:" (e.g. "orders", "customer"). Never add dots, parentheses, or SQL syntax.

Available operation types:
- filter: {"type":"filter","conditions":[{"column":"...","op":"==|!=|<|<=|>|>=","value":<str|int|float>}],"combine":"and"}
- group_by: {"type":"group_by","columns":[...],"aggregations":[{"agg":"sum|mean|min|max|len","column":"...","alias":"..."}]}
- sort: {"type":"sort","columns":[{"name":"...","descending":true}]}
- select: {"type":"select","columns":[...]}
- join: {"type":"join","table":"...","on":"...","how":"inner"}
- head: {"type":"head","n":<int>}

Rules:
- source_table = the first table you query (bare name, no prefix).
- Use only column names that exist in the schema.
- Output only the JSON object."""


_FEWSHOT: list[tuple[dict, str, str]] = [
    (
        {"orders": {"columns": {"o_orderkey": "Int64", "o_totalprice": "Float64", "o_orderstatus": "Utf8"}, "n_rows": 1500000}},
        "Return the 10 orders with the highest total price.",
        '{"source_table":"orders","operations":[{"type":"sort","columns":[{"name":"o_totalprice","descending":true}]},{"type":"head","n":10}]}',
    ),
    (
        {"lineitem": {"columns": {"l_orderkey": "Int64", "l_extendedprice": "Float64", "l_shipdate": "Date"}, "n_rows": 6000000}},
        "Count lineitems per order, sorted by count descending.",
        '{"source_table":"lineitem","operations":[{"type":"group_by","columns":["l_orderkey"],"aggregations":[{"agg":"len","column":"l_orderkey","alias":"count"}]},{"type":"sort","columns":[{"name":"count","descending":true}]}]}',
    ),
    (
        {
            "customer": {"columns": {"c_custkey": "Int64", "c_name": "Utf8", "c_nationkey": "Int64"}, "n_rows": 150000},
            "nation": {"columns": {"n_nationkey": "Int64", "n_name": "Utf8"}, "n_rows": 25},
        },
        "Return customer name and nation name for all customers.",
        '{"source_table":"customer","operations":[{"type":"join","table":"nation","on":"","left_on":"c_nationkey","right_on":"n_nationkey","how":"inner"},{"type":"select","columns":["c_name","n_name"]}]}',
    ),
]


def _format_schema(tables: dict) -> str:
    lines = ["Datasets:"]
    for name, meta in tables.items():
        cols = meta.get("columns") if isinstance(meta, dict) else meta
        n_rows = meta.get("n_rows") if isinstance(meta, dict) else None
        header = f"- {name}" + (f" ({n_rows} rows)" if n_rows else "")
        lines.append(header)
        for col, dtype in (cols.items() if isinstance(cols, dict) else {}.items()):
            lines.append(f"    {col}: {dtype}")
    return "\n".join(lines)


class JsonQueryGenerator:
    """Wraps GemmaModel to generate QueryPlan JSON instead of raw Polars code."""

    def __init__(self, model_name: str = "google/gemma-4-E2B-it"):
        # Import here to allow importing this module without GPU deps
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self._outlines_model = None

    def _ensure_outlines(self):
        if self._outlines_model is None:
            import outlines
            self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)

    def _build_prompt(self, tables: dict, question: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for fs_tables, fs_q, fs_a in _FEWSHOT:
            messages.append({"role": "user", "content": f"{_format_schema(fs_tables)}\n\nQuestion: {fs_q}"})
            messages.append({"role": "assistant", "content": fs_a})
        messages.append({"role": "user", "content": f"{_format_schema(tables)}\n\nQuestion: {question}"})
        return messages

    def _greedy(self, prompt: str, max_new_tokens: int = 768) -> str:
        import torch
        with torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            return self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first {...} block out of free-form text."""
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
        return text[start:]  # truncated — return what we have, let validator report it

    def generate(self, tables: dict, question: str) -> QueryPlan:
        """JSON generation with constrained-then-greedy fallback.

        L1: outlines.Generator with QueryPlan as output_type — builds a JSON-schema
            logits processor so tokens are constrained to valid QueryPlan JSON.
        L2: if L1 produces invalid JSON (e.g. truncation from llguidance), fall back
            to unconstrained greedy decoding and extract the first {...} block.
        """
        import outlines

        self._ensure_outlines()
        messages = self._build_prompt(tables, question)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # L1 — constrained
        try:
            generator = outlines.Generator(self._outlines_model, QueryPlan)
            result: str = generator(prompt, max_new_tokens=768)
            return QueryPlan.model_validate_json(result)
        except Exception as e:
            print(f"[json_gen] constrained failed ({type(e).__name__}: {e}), falling back to greedy")

        # L2 — greedy + JSON extraction
        raw = self._greedy(prompt, max_new_tokens=768)
        return QueryPlan.model_validate_json(self._extract_json(raw))

    def generate_and_convert(self, tables: dict, question: str) -> tuple[QueryPlan, str]:
        """Generate a QueryPlan and convert it to Polars code. Returns (plan, code)."""
        plan = self.generate(tables, question)
        code = json_to_polars(plan)
        return plan, code


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python json_query_generator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test json_to_polars without loading the model
    sample_plan = QueryPlan(
        source_table="sales",
        operations=[
            FilterOp(
                type="filter",
                conditions=[
                    Condition(column="region", op=CompareOp.eq, value="Europe"),
                    Condition(column="revenue", op=CompareOp.gt, value=100),
                ],
                combine="and",
            ),
            SortOp(
                type="sort",
                columns=[SortColumn(name="revenue", descending=True)],
            ),
            HeadOp(type="head", n=10),
        ],
    )

    print("=== QueryPlan JSON ===")
    print(json.dumps(sample_plan.model_dump(), indent=2))
    print()
    print("=== Generated Polars code ===")
    print(json_to_polars(sample_plan))
    print()

    # Test round-trip from raw dict
    raw = {
        "source_table": "orders",
        "operations": [
            {
                "type": "group_by",
                "columns": ["user_id"],
                "aggregations": [
                    {"agg": "sum", "column": "amount", "alias": "total"},
                    {"agg": "len", "column": "*", "alias": "count"},
                ],
            },
            {"type": "sort", "columns": [{"name": "total", "descending": True}]},
            {"type": "head", "n": 5},
        ],
    }
    print("=== Round-trip from dict ===")
    print(json_to_polars(raw))

    # Only load model if --model flag passed
    if "--model" in sys.argv:
        tables = {
            "sales": {
                "columns": {"product": "Utf8", "revenue": "Float64", "region": "Utf8"},
                "n_rows": 1000,
            }
        }
        question = "Return all sales in Europe with revenue above 100, sorted by revenue descending."
        gen = JsonQueryGenerator()
        plan, code = gen.generate_and_convert(tables, question)
        print("\n=== Model-generated plan ===")
        print(json.dumps(plan.model_dump(), indent=2))
        print("\n=== Converted Polars code ===")
        print(code)
