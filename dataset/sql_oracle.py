"""Exécute du SQL contre un dict de DataFrames Polars via duckdb.

Sert d'oracle: chaque (question, SQL) de Spider passe par ici pour obtenir
le DataFrame de référence que le code Polars généré devra reproduire.
"""
from dataclasses import dataclass

import duckdb
import polars as pl


@dataclass
class SqlExecutionResult:
    success: bool
    result: pl.DataFrame | None
    error: str | None


def execute_sql(sql: str, tables: dict[str, pl.DataFrame]) -> SqlExecutionResult:
    """Exécute `sql` avec duckdb. Les clés de `tables` deviennent des noms
    de table accessibles depuis la requête. Case-insensitive par défaut.
    """
    con = duckdb.connect()
    try:
        for name, df in tables.items():
            con.register(name, df.to_arrow())
        try:
            arrow_tbl = con.execute(sql).fetch_arrow_table()
        except Exception as exc:
            return SqlExecutionResult(False, None, f"{type(exc).__name__}: {exc}")
        df = pl.from_arrow(arrow_tbl)
        if isinstance(df, pl.Series):
            df = df.to_frame()
        return SqlExecutionResult(True, df, None)
    finally:
        con.close()
