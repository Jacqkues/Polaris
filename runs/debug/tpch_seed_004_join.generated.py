result = (
    customer
    .join(nation, left_on="c_nationkey", right_on="n_nationkey")
    .select([pl.col("c_name"), pl.col("n_name").alias("nation")])
    .sort("c_name")
    .head(20)
)