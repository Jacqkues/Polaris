result = (
    orders
    .sort("o_totalprice", descending=True)
    .head(10)
    .select(["o_orderkey", "o_totalprice", "o_orderdate"])
)