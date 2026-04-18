"""Convertit les bases SQLite de Spider en parquets locaux.

Prérequis: télécharger le zip Spider depuis https://yale-lily.github.io/spider
et l'extraire. On attend un dossier contenant `database/<db_id>/<db_id>.sqlite`
pour chaque base.

Usage:
  python -m dataset.spider_setup \\
      --spider-dir /path/to/spider_data \\
      --dbs concert_singer pets_1 \\
      --out data/spider
"""
import argparse
import json
from pathlib import Path

import duckdb

DEFAULT_DBS = ["concert_singer", "pets_1"]


def convert_db(sqlite_path: Path, out_dir: Path) -> dict[str, int]:
    """Convertit chaque table d'un .sqlite en parquet. Retourne {table: n_rows}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL sqlite")
    con.execute("LOAD sqlite")
    con.execute(f"ATTACH '{sqlite_path}' AS src (TYPE sqlite, READ_ONLY)")

    tables = [
        row[0]
        for row in con.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    row_counts: dict[str, int] = {}
    for table in tables:
        out_path = out_dir / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT * FROM src.\"{table}\") TO '{out_path}' (FORMAT PARQUET)"
        )
        n_rows = con.execute(f'SELECT COUNT(*) FROM src."{table}"').fetchone()[0]
        row_counts[table] = n_rows
    con.close()
    return row_counts


def main(spider_dir: Path, dbs: list[str], out_dir: Path) -> int:
    db_root = spider_dir / "database"
    if not db_root.is_dir():
        print(f"ERROR: {db_root} introuvable. Spider est-il extrait à {spider_dir}?")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Conversion de {len(dbs)} base(s) → {out_dir}\n")

    manifest: dict[str, dict[str, int]] = {}
    for db_id in dbs:
        sqlite_path = db_root / db_id / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            print(f"  SKIP  {db_id}: {sqlite_path} introuvable")
            continue

        db_out = out_dir / db_id
        row_counts = convert_db(sqlite_path, db_out)
        manifest[db_id] = row_counts
        total = sum(row_counts.values())
        print(f"  OK    {db_id:25s}  {len(row_counts)} tables, {total:,} rows → {db_out}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", type=Path, required=True,
                   help="Dossier Spider extrait (contient database/)")
    p.add_argument("--dbs", nargs="+", default=DEFAULT_DBS,
                   help="Liste des db_id à convertir")
    p.add_argument("--out", type=Path, default=Path("data/spider"))
    args = p.parse_args()
    raise SystemExit(main(args.spider_dir, args.dbs, args.out))
