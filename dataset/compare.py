"""Column-order-invariant DataFrame comparison with diagnostic reason codes.

Used by the benchmark to tell us *why* a generated output mismatched the
reference — not just that the hash differed.
"""
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

FLOAT_DTYPES = (pl.Float32, pl.Float64)
Reason = Literal["match", "empty_actual", "column_set", "row_count", "content"]


@dataclass
class ComparisonResult:
    match: bool
    reason: Reason
    actual_n_rows: int
    expected_n_rows: int
    actual_columns: list[str]
    expected_columns: list[str]
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    actual_preview: str = ""
    expected_preview: str = ""
    detail: str = ""

    def summary(self) -> str:
        if self.match:
            return "match"
        if self.reason == "column_set":
            return (
                f"column_set: missing={self.missing_columns}  extra={self.extra_columns}"
            )
        if self.reason == "row_count":
            return f"row_count: actual={self.actual_n_rows} expected={self.expected_n_rows}"
        if self.reason == "content":
            return f"content: {self.actual_n_rows} rows match in count but values/order differ"
        if self.reason == "empty_actual":
            return "empty_actual: generated code produced an empty DataFrame"
        return self.reason


def _preview(df: pl.DataFrame, n: int = 5) -> str:
    if df.is_empty():
        return "<empty>"
    with pl.Config(
        tbl_rows=n,
        tbl_cols=-1,
        fmt_str_lengths=50,
        tbl_width_chars=160,
    ):
        return str(df.head(n))


def _normalize(df: pl.DataFrame, float_precision: int) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col(name).round(float_precision) if dtype in FLOAT_DTYPES else pl.col(name)
            for name, dtype in df.schema.items()
        ]
    )
    sortable = [c for c, dt in df.schema.items() if not isinstance(dt, (pl.List, pl.Struct))]
    if sortable:
        df = df.sort(sortable)
    return df


def compare_dataframes(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    float_precision: int = 6,
) -> ComparisonResult:
    base = dict(
        actual_n_rows=actual.height,
        expected_n_rows=expected.height,
        actual_columns=list(actual.columns),
        expected_columns=list(expected.columns),
        actual_preview=_preview(actual),
        expected_preview=_preview(expected),
    )

    actual_cols = set(actual.columns)
    expected_cols = set(expected.columns)
    missing = sorted(expected_cols - actual_cols)
    extra = sorted(actual_cols - expected_cols)

    if missing or extra:
        return ComparisonResult(
            match=False,
            reason="column_set",
            missing_columns=missing,
            extra_columns=extra,
            **base,
        )

    if actual.height == 0:
        return ComparisonResult(match=False, reason="empty_actual", **base)

    if actual.height != expected.height:
        return ComparisonResult(match=False, reason="row_count", **base)

    # Align to expected column order, then normalize (round floats + sort rows).
    aligned = actual.select(expected.columns)
    a_norm = _normalize(aligned, float_precision)
    e_norm = _normalize(expected, float_precision)

    if a_norm.equals(e_norm):
        return ComparisonResult(match=True, reason="match", **base)

    # Find first differing column for a more specific hint.
    differing = [c for c in expected.columns if not a_norm[c].equals(e_norm[c])]
    detail = f"columns with differing values: {differing}" if differing else ""
    return ComparisonResult(match=False, reason="content", detail=detail, **base)
