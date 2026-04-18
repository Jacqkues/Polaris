result = (
    orders
    .filter(pl.col("o_orderdate").is_between(pl.date(1995, 1, 1), pl.date(1995, 3, 31)))
    .select(pl.len().alias("q1_1995_orders"))
)