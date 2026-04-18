"""Execute each seed against real TPC-H data, compute expected outputs, and
emit the canonical JSONL dataset.

Usage: python -m dataset.build_seeds
"""
import argparse
import json
from pathlib import Path

from dataset.executor import execute_code, load_tpch
from dataset.hashing import hash_dataframe
from dataset.schema import TPCH_TABLES, DatasetRecord, TableSchema
from dataset.seeds import SEEDS


def build_record(seed: dict, tables: dict, table_row_counts: dict[str, int]) -> tuple[DatasetRecord | None, str | None]:
    result = execute_code(seed["reference_code"], tables)
    if not result.success:
        return None, result.error

    df = result.result
    used_tables = {
        name: TableSchema(columns=TPCH_TABLES[name], n_rows=table_row_counts[name])
        for name in seed["tables_used"]
    }
    record = DatasetRecord(
        id=seed["id"],
        tables=used_tables,
        question=seed["question"],
        reference_code=seed["reference_code"],
        tags=seed["tags"],
        difficulty=seed["difficulty"],
        expected_output_hash=hash_dataframe(df),
        expected_n_rows=df.height,
        expected_columns=df.columns,
    )
    return record, None


def main(data_dir: Path, out_path: Path) -> int:
    tables = load_tpch(data_dir)
    if not tables:
        print(f"No parquet files found in {data_dir}. Run: python -m dataset.gen_tpch first.")
        return 1
    row_counts = {name: df.height for name, df in tables.items()}
    print(f"Loaded {len(tables)} TPC-H tables: " + ", ".join(f"{k}={v}" for k, v in row_counts.items()))
    print()

    passed, failed = 0, 0
    records: list[DatasetRecord] = []
    for seed in SEEDS:
        record, error = build_record(seed, tables, row_counts)
        if error:
            print(f"  FAIL  {seed['id']}: {error}")
            failed += 1
            continue
        records.append(record)
        print(
            f"  OK    {seed['id']:35s}  "
            f"rows={record.expected_n_rows:>6d}  cols={len(record.expected_columns)}"
        )
        passed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    print(f"\n{passed} passed, {failed} failed → {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/tpch"))
    p.add_argument("--out", type=Path, default=Path("data/seeds.jsonl"))
    args = p.parse_args()
    raise SystemExit(main(args.data, args.out))
