"""Charge un subset du dataset Spider (questions + SQL) depuis HuggingFace,
et les associe aux parquets locaux produits par `spider_setup.py`.
"""
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from datasets import load_dataset

SPIDER_HF_ID = "xlangai/spider"


@dataclass
class SpiderExample:
    id: str                               # "spider_<db_id>_<idx>"
    db_id: str
    question: str
    sql: str
    tables: dict[str, pl.DataFrame]       # données réelles
    schemas: dict[str, dict[str, str]]    # {table: {col: dtype}}

    def schemas_for_prompt(self) -> dict[str, dict[str, str]]:
        return self.schemas

    def table_row_counts(self) -> dict[str, int]:
        return {name: df.height for name, df in self.tables.items()}


def _polars_dtype_name(dtype: pl.DataType) -> str:
    return str(dtype)


def load_db_tables(spider_out: Path, db_id: str) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, str]]]:
    db_dir = spider_out / db_id
    if not db_dir.is_dir():
        raise FileNotFoundError(
            f"{db_dir} introuvable. Lance `python -m dataset.spider_setup` d'abord."
        )
    tables: dict[str, pl.DataFrame] = {}
    schemas: dict[str, dict[str, str]] = {}
    for parquet in sorted(db_dir.glob("*.parquet")):
        df = pl.read_parquet(parquet)
        tables[parquet.stem] = df
        schemas[parquet.stem] = {name: _polars_dtype_name(dt) for name, dt in df.schema.items()}
    if not tables:
        raise FileNotFoundError(f"Aucun parquet dans {db_dir}")
    return tables, schemas


def load_spider_subset(
    spider_out: Path,
    db_whitelist: list[str] | None = None,
    split: str = "train",
    limit: int | None = None,
) -> list[SpiderExample]:
    """Charge les questions Spider depuis HF, filtre aux bases whitelistées
    (celles effectivement présentes en parquet local), hydrate chaque exemple
    avec ses DataFrames."""
    ds = load_dataset(SPIDER_HF_ID, split=split)

    available_dbs = {p.name for p in spider_out.iterdir() if p.is_dir()}
    if db_whitelist is not None:
        available_dbs &= set(db_whitelist)
    if not available_dbs:
        raise RuntimeError(
            f"Aucune base locale dans {spider_out}. Lance spider_setup d'abord."
        )

    # Cache par db_id pour éviter de relire les parquets N fois.
    cache: dict[str, tuple[dict[str, pl.DataFrame], dict[str, dict[str, str]]]] = {}

    examples: list[SpiderExample] = []
    counters: dict[str, int] = {}
    for row in ds:
        db_id = row["db_id"]
        if db_id not in available_dbs:
            continue
        if db_id not in cache:
            cache[db_id] = load_db_tables(spider_out, db_id)
        tables, schemas = cache[db_id]
        idx = counters.get(db_id, 0)
        counters[db_id] = idx + 1
        examples.append(
            SpiderExample(
                id=f"spider_{db_id}_{idx:04d}",
                db_id=db_id,
                question=row["question"],
                sql=row["query"],
                tables=tables,
                schemas=schemas,
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples
