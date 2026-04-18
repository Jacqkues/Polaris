result = (
    customer
    .filter(pl.col("c_mktsegment") == "BUILDING")
    .select(["c_custkey", "c_name"])
)