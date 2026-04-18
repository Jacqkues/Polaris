algerian_parts = (
    partsupp
    .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    .join(nation, left_on="s_nationkey", right_on="n_nationkey")
    .filter(pl.col("n_name") == "ALGERIA")
    .select("ps_partkey")
)
result = (
    part
    .join(algerian_parts, left_on="p_partkey", right_on="ps_partkey", how="semi")
    .select(["p_partkey", "p_name"])
    .sort("p_partkey")
    .head(20)
)