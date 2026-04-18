"""Prompt & few-shot definitions for the Gemma 4 E2B benchmark.

Kept in a separate module so we can iterate on prompt engineering (cheat sheet
rules, few-shot examples, schema format) without touching benchmark_gemma.py's
plumbing. Easy to A/B test by swapping imports or creating prompt_v2.py etc.

Exports:
  - SYSTEM_PROMPT : str      — Polars cheat sheet + rules (A1)
  - FEWSHOT       : list     — chat-format examples (A2)
  - format_schema : callable — readable typed schema block (A3)
  - format_user_turn : callable — schema + question for a user turn
"""


SYSTEM_PROMPT = """Return only valid Python Polars code (no markdown fences, no prose).

Rules:
- Assign the final DataFrame to `result`.
- The provided DataFrames are already in scope by name — do not recreate them with pl.DataFrame() or pl.read_*.
- Use Polars syntax, NOT pandas: `group_by` (not `groupby`), `pl.col("x")` for columns (never bare strings inside expressions or `.col_name` attribute access).

Modern Polars API (use exactly these forms — common mistakes to avoid):
- df.with_columns(...)          NOT df.with_column(...)
- df.sort("x", descending=True) NOT pl.desc("x")   (pl.desc does not exist)
- df.group_by("x").agg(...)     NOT df.agg(...) directly on a DataFrame
- expr.str.contains("...")      NOT expr.contains(...)
- df.height  or  pl.len()       NOT df.len / df.length
- expr.rank(method="dense")     NOT expr.dense(...)
- df.filter(pl.col("x") > 10)   NOT df[df["x"] > 10]  (no boolean indexing)
- .join(other, on="k")          or  .join(other, left_on="a", right_on="b")

Dates:
- `.dt.year()`, `.dt.month()`, `.dt.day()` for parts
- `pl.date(Y, M, D)` for literals
- `.is_between(a, b)` for ranges

Aggregation:
- `pl.len()` for row count; use `.alias("name")` to rename outputs.
- For top-N per group use a window + filter, or sort+group_by+head.

Output shape (critical — the output columns must match what the question asks for):
- After a join, ALWAYS end with `.select([...])` listing only the columns the
  question actually mentions. Never leave all joined columns in the output.
- For an ungrouped count (the question asks "how many X" without partition),
  use `.select(pl.len().alias(...))`. Do NOT use `.group_by(pl.lit(None)).agg(...)`:
  that creates a spurious "literal" column in the output.
- ALWAYS add `.alias(...)` to aggregation expressions. An un-aliased
  `pl.col("x").n_unique()` keeps the column name "x", which is almost never
  what the question asks for. Same for `.sum()`, `.mean()`, `.max()`, etc.
- When grouping by an expression (not a plain column), alias the expression:
  `.group_by(pl.col("d").dt.year().alias("year"))`. Otherwise the group column
  is named after the raw expression, not after the concept it represents."""


FEWSHOT: list[tuple[dict, str, str]] = [
    (
        {
            "sales": {
                "columns": {"product": "Utf8", "revenue": "Float64", "region": "Utf8"},
                "n_rows": 1000,
            }
        },
        "Return all sales in Europe with revenue above 100, sorted by revenue descending.",
        'result = (\n'
        '    sales\n'
        '    .filter((pl.col("region") == "Europe") & (pl.col("revenue") > 100))\n'
        '    .sort("revenue", descending=True)\n'
        ')',
    ),
    (
        {
            "events": {
                "columns": {"user_id": "Int64", "event_type": "Utf8", "ts": "Datetime"},
                "n_rows": 50000,
            }
        },
        "Count events per event_type for 2024, sorted by count descending.",
        'result = (\n'
        '    events\n'
        '    .filter(pl.col("ts").dt.year() == 2024)\n'
        '    .group_by("event_type")\n'
        '    .agg(pl.len().alias("count"))\n'
        '    .sort("count", descending=True)\n'
        ')',
    ),
    (
        {
            "users": {"columns": {"user_id": "Int64", "name": "Utf8"}, "n_rows": 500},
            "orders": {
                "columns": {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"},
                "n_rows": 10000,
            },
        },
        "For each user, return the total amount of their orders, top 5.",
        'result = (\n'
        '    users\n'
        '    .join(orders, on="user_id")\n'
        '    .group_by("name")\n'
        '    .agg(pl.col("amount").sum().alias("total"))\n'
        '    .sort("total", descending=True)\n'
        '    .head(5)\n'
        ')',
    ),
    (
        {
            "products": {
                "columns": {"product_id": "Int64", "category": "Utf8", "price": "Float64"},
                "n_rows": 2000,
            }
        },
        "Rank products by price within each category, highest first, and keep the top 2 per category.",
        'result = (\n'
        '    products\n'
        '    .with_columns(\n'
        '        pl.col("price").rank(method="dense", descending=True).over("category").alias("rnk")\n'
        '    )\n'
        '    .filter(pl.col("rnk") <= 2)\n'
        '    .sort(["category", "rnk"])\n'
        ')',
    ),
]


def format_schema(tables: dict) -> str:
    """Readable schema block with n_rows per table. Falls back gracefully if a
    record lacks n_rows (defensive for future dataset changes)."""
    lines = ["Datasets:"]
    for name, meta in tables.items():
        cols = meta.get("columns") if isinstance(meta, dict) else None
        n_rows = meta.get("n_rows") if isinstance(meta, dict) else None
        if cols is None:
            cols = meta
            n_rows = None
        header = f"- {name}" + (f" ({n_rows} rows)" if n_rows is not None else "")
        lines.append(header)
        for col, dtype in cols.items():
            lines.append(f"    {col}: {dtype}")
    return "\n".join(lines)


def format_user_turn(tables: dict, question: str) -> str:
    return f"{format_schema(tables)}\n\nQuestion: {question}"
