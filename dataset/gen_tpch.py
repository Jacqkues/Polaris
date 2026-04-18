"""Generate TPC-H Parquet files via duckdb's dbgen extension.

Usage: python -m dataset.gen_tpch --sf 0.01 --out data/tpch
"""
import argparse
from pathlib import Path

import duckdb

TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]


def main(scale_factor: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale_factor})")
    print(f"Generating TPC-H SF={scale_factor} → {output_dir}\n")
    for table in TABLES:
        out = output_dir / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET)")
        n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:10s}  {n_rows:>10,d} rows  →  {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sf", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=Path("data/tpch"))
    args = p.parse_args()
    main(args.sf, args.out)
