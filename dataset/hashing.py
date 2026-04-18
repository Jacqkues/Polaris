import hashlib

import polars as pl


FLOAT_DTYPES = (pl.Float32, pl.Float64)


def hash_dataframe(df: pl.DataFrame, float_precision: int = 6) -> str:
    """Deterministic hash of a DataFrame's content.

    Rounds floats (to tolerate FP drift across platforms), sorts rows, writes
    to CSV, then sha256. Column order is preserved because it is semantic.
    """
    if df.is_empty():
        fingerprint = b"<empty>:" + ",".join(df.columns).encode()
        return hashlib.sha256(fingerprint).hexdigest()

    df_norm = df.with_columns(
        [
            pl.col(name).round(float_precision) if dtype in FLOAT_DTYPES else pl.col(name)
            for name, dtype in df.schema.items()
        ]
    ).sort(df.columns)
    payload = df_norm.write_csv()
    return hashlib.sha256(payload.encode()).hexdigest()
