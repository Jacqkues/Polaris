result = (
    lineitem
    .join(orders, left_on="l_orderkey", right_on="o_orderkey")
    .join(customer, left_on="o_custkey", right_on="c_custkey")
    .join(nation, left_on="c_nationkey", right_on="n_nationkey")
    .filter(pl.col("o_orderdate").is_between(pl.date(1994, 1, 1), pl.date(1994, 12, 31)))
    .with_columns((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue"))
    .group_by("n_name")
    .agg(pl.col("revenue").sum())
    .sort("revenue", descending=True)
)