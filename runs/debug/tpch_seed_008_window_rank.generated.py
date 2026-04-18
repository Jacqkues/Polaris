result = (
    customer
    .with_columns(
        pl.col("c_acctbal")
        .rank(method="dense", descending=True)
        .over("c_nationkey")
        .alias("rank_in_nation")
    )
    .filter(pl.col("rank_in_nation") <= 3)
    .select(["c_custkey", "c_name", "c_nationkey", "c_acctbal", "rank_in_nation"])
    .sort(["c_nationkey", "rank_in_nation", "c_custkey"])
)