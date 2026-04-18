"""Construit le dataset SFT candidat à partir de Spider.

Pipeline: charge un subset Spider → exécute chaque SQL via duckdb → garde
les exemples qui retournent un DataFrame non-vide → écrit:
  - data/spider_sft.jsonl         (un record par exemple validé)
  - data/spider_expected/<id>.parquet  (DataFrame oracle)

Le champ `polars_code` est laissé à None — à remplir dans une étape
ultérieure (LLM-based génération + validation via dataset/compare.py).

Usage:
  python -m dataset.build_sft --limit 200
  python -m dataset.build_sft --dbs concert_singer pets_1 --split train
"""
import argparse
import json
from pathlib import Path

from dataset.hashing import hash_dataframe
from dataset.spider_loader import load_spider_subset
from dataset.sql_oracle import execute_sql


def main(
    spider_out: Path,
    dbs: list[str] | None,
    split: str,
    limit: int | None,
    out_jsonl: Path,
    expected_dir: Path,
) -> int:
    examples = load_spider_subset(spider_out, db_whitelist=dbs, split=split, limit=limit)
    print(f"Chargé {len(examples)} exemples Spider ({split})\n")

    expected_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ok, sql_fail, empty = 0, 0, 0
    records: list[dict] = []

    for ex in examples:
        res = execute_sql(ex.sql, ex.tables)
        if not res.success:
            sql_fail += 1
            print(f"  SQL-FAIL  {ex.id}  {res.error}")
            continue
        if res.result.is_empty():
            empty += 1
            continue

        parquet_path = expected_dir / f"{ex.id}.parquet"
        res.result.write_parquet(parquet_path)

        records.append({
            "id": ex.id,
            "db_id": ex.db_id,
            "question": ex.question,
            "sql": ex.sql,
            "tables": {
                name: {"columns": cols, "n_rows": ex.tables[name].height}
                for name, cols in ex.schemas.items()
            },
            "polars_code": None,
            "expected_output_hash": hash_dataframe(res.result),
            "expected_n_rows": res.result.height,
            "expected_columns": res.result.columns,
        })
        ok += 1

    with out_jsonl.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nRésumé: {ok} OK  |  {empty} empty  |  {sql_fail} SQL fail")
    print(f"JSONL    → {out_jsonl}")
    print(f"Parquets → {expected_dir}/ ({ok} fichiers)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spider-out", type=Path, default=Path("data/spider"),
                   help="Dossier contenant les parquets produits par spider_setup")
    p.add_argument("--dbs", nargs="+", default=None,
                   help="Subset de db_id; défaut = toutes les bases présentes localement")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("data/spider_sft.jsonl"))
    p.add_argument("--expected", type=Path, default=Path("data/spider_expected"))
    args = p.parse_args()
    raise SystemExit(main(
        args.spider_out, args.dbs, args.split, args.limit, args.out, args.expected
    ))
