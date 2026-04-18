result = (
    customer
    .filter(pl.col("c_mktsegment") == "BUILDING")
    .join(orders, left_on="c_custkey", right_on="o_custkey")
    .filter(pl.col("o_orderdate") < pl.date(1995, 3, 15))
    .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    .filter(pl.col("l_shipdate") > pl.date(1995, 3, 15))
    .with_columns((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue"))
    .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
    .agg(pl.col("revenue").sum())
    .sort(["revenue", "o_orderdate"], descending=[True, False])
    .head(10)
)