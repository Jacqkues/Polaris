from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class ExecutionResult:
    success: bool
    result: pl.DataFrame | None
    error: str | None


def load_tpch(data_dir: Path) -> dict[str, pl.DataFrame]:
    tables = {}
    for path in sorted(data_dir.glob("*.parquet")):
        tables[path.stem] = pl.read_parquet(path)
    return tables


def execute_code(code: str, tables: dict[str, pl.DataFrame]) -> ExecutionResult:
    """Execute a generated Polars snippet. Contract: the snippet must assign
    the final DataFrame to `result`. Returns ExecutionResult with the
    collected DataFrame (LazyFrame is collected automatically)."""
    env: dict = {"pl": pl, **tables}
    try:
        exec(code, env)  # noqa: S102 — trusted dev input; sandbox later
    except Exception as exc:
        return ExecutionResult(False, None, f"{type(exc).__name__}: {exc}")

    result = env.get("result")
    if result is None:
        return ExecutionResult(False, None, "No `result` variable assigned")
    if isinstance(result, pl.LazyFrame):
        try:
            result = result.collect()
        except Exception as exc:
            return ExecutionResult(False, None, f"collect() failed: {exc}")
    if not isinstance(result, pl.DataFrame):
        return ExecutionResult(
            False, None, f"`result` is {type(result).__name__}, expected DataFrame"
        )
    return ExecutionResult(True, result, None)
