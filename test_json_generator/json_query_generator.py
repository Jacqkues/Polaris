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
    conditions: list[Condition] = Field(min_length=1)
    combine: Literal["and", "or"] = "and"


class Aggregation(BaseModel):
    agg: Literal["sum", "mean", "min", "max", "len", "n_unique", "first", "last"]
    column: str  # "*" means pl.len() (row count)
    alias: str


class GroupByOp(BaseModel):
    type: Literal["group_by"]
    columns: list[str] = Field(min_length=1)
    aggregations: list[Aggregation] = Field(min_length=1)


class SortColumn(BaseModel):
    name: str
    descending: bool = False


class SortOp(BaseModel):
    type: Literal["sort"]
    columns: list[SortColumn] = Field(min_length=1)


class SelectOp(BaseModel):
    type: Literal["select"]
    columns: list[str] = Field(min_length=1)


class JoinOp(BaseModel):
    type: Literal["join"]
    table: str
    on: str | None = None          # shared key name
    left_on: str | None = None
    right_on: str | None = None
    how: Literal["inner", "left", "outer"] = "inner"


class WithColumnExpr(BaseModel):
    column: str
    expr: Literal["rank"]
    method: Literal["dense", "ordinal", "average", "min", "max"] = "dense"
    descending: bool = True
    over: str | None = None    # window partition column
    alias: str


class WithColumnsOp(BaseModel):
    type: Literal["with_columns"]
    expressions: list[WithColumnExpr] = Field(min_length=1)


class HeadOp(BaseModel):
    type: Literal["head"]
    n: int = Field(ge=1)


Operation = Annotated[
    Union[FilterOp, GroupByOp, SortOp, SelectOp, JoinOp, WithColumnsOp, HeadOp],
    Field(discriminator="type"),
]


class QueryPlan(BaseModel):
    source_table: str
    operations: list[Operation] = Field(min_length=1)


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
        if e.over:
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

_SYSTEM_PROMPT = """You are a data query planner. Given a dataset schema and a question, output a JSON query plan — nothing else.

The JSON must match this structure exactly:
{
  "source_table": "<table name>",
  "operations": [ ... ]
}

Available operation types:
- filter: { "type": "filter", "conditions": [{"column": "...", "op": "==|!=|<|<=|>|>=", "value": <str|int|float|bool>}], "combine": "and"|"or" }
- group_by: { "type": "group_by", "columns": [...], "aggregations": [{"agg": "sum|mean|min|max|len|n_unique|first|last", "column": "<col or *>", "alias": "..."}] }
- sort: { "type": "sort", "columns": [{"name": "...", "descending": true|false}] }
- select: { "type": "select", "columns": [...] }
- join: { "type": "join", "table": "...", "on": "...", "how": "inner|left|outer" }
- with_columns: { "type": "with_columns", "expressions": [{"column": "...", "expr": "rank", "method": "dense", "descending": true, "over": "<col>", "alias": "..."}] }
- head: { "type": "head", "n": <int> }

Rules:
- Use only column names that exist in the schema.
- Output only the JSON object, no markdown fences, no explanation.
- Operations execute in the order listed."""


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
        schema_block = _format_schema(tables)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{schema_block}\n\nQuestion: {question}"},
        ]

    def generate(self, tables: dict, question: str) -> QueryPlan:
        """Constrained JSON generation — output is guaranteed to match QueryPlan schema.

        Uses outlines.Generator with the Pydantic model as output_type, which
        builds a JSON-schema logits processor so the decoder can only emit tokens
        that form valid QueryPlan JSON.  The generator returns a raw JSON string;
        we parse it into a QueryPlan with model_validate_json.
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
        # outlines v1 API: Generator(model, output_type) → callable that returns str
        generator = outlines.Generator(self._outlines_model, QueryPlan)
        result: str = generator(prompt, max_tokens=512)
        return QueryPlan.model_validate_json(result)

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
