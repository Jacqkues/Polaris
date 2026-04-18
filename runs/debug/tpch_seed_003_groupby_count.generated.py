result = (
    orders
    .group_by("o_orderstatus")
    .agg(pl.len().alias("n_orders"))
    .sort("o_orderstatus")
)