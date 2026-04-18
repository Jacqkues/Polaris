result = (
    customer
    .filter(pl.col("c_name").str.contains("Customer#0000001"))
    .select(["c_custkey", "c_name"])
    .sort("c_custkey")
)