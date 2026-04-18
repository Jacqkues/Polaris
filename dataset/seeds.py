"""Hand-crafted seed examples for TPC-H.

Each seed is the template we'll generalize from: one question + one reference
Polars snippet that, when executed, assigns the canonical DataFrame to
`result`. Seeds span the coverage matrix (filter, sort, groupby, join, window,
string, date, multi-join, semi-join).
"""

SEEDS: list[dict] = [
    {
        "id": "tpch_seed_001_filter",
        "tags": ["filter", "select"],
        "difficulty": 1,
        "tables_used": ["customer"],
        "question": "List the customer keys and names of customers in the BUILDING market segment.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .filter(pl.col("c_mktsegment") == "BUILDING")\n'
            '    .select(["c_custkey", "c_name"])\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_002_sort_limit",
        "tags": ["sort", "limit", "select"],
        "difficulty": 1,
        "tables_used": ["orders"],
        "question": "Return the 10 orders with the highest total price, showing order key, total price, and order date.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .sort("o_totalprice", descending=True)\n'
            '    .head(10)\n'
            '    .select(["o_orderkey", "o_totalprice", "o_orderdate"])\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_003_groupby_count",
        "tags": ["groupby", "agg"],
        "difficulty": 1,
        "tables_used": ["orders"],
        "question": "Count the number of orders per order status, sorted alphabetically by status.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .group_by("o_orderstatus")\n'
            '    .agg(pl.len().alias("n_orders"))\n'
            '    .sort("o_orderstatus")\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_004_join",
        "tags": ["join", "select"],
        "difficulty": 2,
        "tables_used": ["customer", "nation"],
        "question": "Return the first 20 customers (by name) with the nation they belong to.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .join(nation, left_on="c_nationkey", right_on="n_nationkey")\n'
            '    .select([pl.col("c_name"), pl.col("n_name").alias("nation")])\n'
            '    .sort("c_name")\n'
            '    .head(20)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_005_revenue_by_nation_1994",
        "tags": ["join", "filter", "groupby", "agg", "date"],
        "difficulty": 4,
        "tables_used": ["lineitem", "orders", "customer", "nation"],
        "question": "Total discounted revenue per nation for orders placed in 1994, sorted descending.",
        "reference_code": (
            'result = (\n'
            '    lineitem\n'
            '    .join(orders, left_on="l_orderkey", right_on="o_orderkey")\n'
            '    .join(customer, left_on="o_custkey", right_on="c_custkey")\n'
            '    .join(nation, left_on="c_nationkey", right_on="n_nationkey")\n'
            '    .filter(pl.col("o_orderdate").is_between(pl.date(1994, 1, 1), pl.date(1994, 12, 31)))\n'
            '    .with_columns((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue"))\n'
            '    .group_by("n_name")\n'
            '    .agg(pl.col("revenue").sum())\n'
            '    .sort("revenue", descending=True)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_006_q1_1995_count",
        "tags": ["filter", "date", "agg"],
        "difficulty": 2,
        "tables_used": ["orders"],
        "question": "How many orders were placed in Q1 1995?",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .filter(pl.col("o_orderdate").is_between(pl.date(1995, 1, 1), pl.date(1995, 3, 31)))\n'
            '    .select(pl.len().alias("q1_1995_orders"))\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_007_string_contains",
        "tags": ["filter", "string"],
        "difficulty": 1,
        "tables_used": ["customer"],
        "question": "Return customer keys and names whose name contains 'Customer#0000001'.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .filter(pl.col("c_name").str.contains("Customer#0000001"))\n'
            '    .select(["c_custkey", "c_name"])\n'
            '    .sort("c_custkey")\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_008_window_rank",
        "tags": ["window", "filter", "sort"],
        "difficulty": 3,
        "tables_used": ["customer"],
        "question": "For each nation, return the top 3 customers by account balance (dense rank).",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .with_columns(\n'
            '        pl.col("c_acctbal")\n'
            '        .rank(method="dense", descending=True)\n'
            '        .over("c_nationkey")\n'
            '        .alias("rank_in_nation")\n'
            '    )\n'
            '    .filter(pl.col("rank_in_nation") <= 3)\n'
            '    .select(["c_custkey", "c_name", "c_nationkey", "c_acctbal", "rank_in_nation"])\n'
            '    .sort(["c_nationkey", "rank_in_nation", "c_custkey"])\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_009_q3_like",
        "tags": ["join", "filter", "groupby", "agg", "sort", "limit"],
        "difficulty": 5,
        "tables_used": ["customer", "orders", "lineitem"],
        "question": (
            "For BUILDING segment customers with orders before 1995-03-15 and line items shipped after "
            "1995-03-15, return the top 10 orders by summed discounted revenue (TPC-H Q3)."
        ),
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .filter(pl.col("c_mktsegment") == "BUILDING")\n'
            '    .join(orders, left_on="c_custkey", right_on="o_custkey")\n'
            '    .filter(pl.col("o_orderdate") < pl.date(1995, 3, 15))\n'
            '    .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")\n'
            '    .filter(pl.col("l_shipdate") > pl.date(1995, 3, 15))\n'
            '    .with_columns((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue"))\n'
            '    .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])\n'
            '    .agg(pl.col("revenue").sum())\n'
            '    .sort(["revenue", "o_orderdate"], descending=[True, False])\n'
            '    .head(10)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_010_semi_join",
        "tags": ["join", "semi_join", "filter"],
        "difficulty": 4,
        "tables_used": ["part", "partsupp", "supplier", "nation"],
        "question": "Return the first 20 parts (by partkey) that have at least one supplier based in ALGERIA.",
        "reference_code": (
            'algerian_parts = (\n'
            '    partsupp\n'
            '    .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")\n'
            '    .join(nation, left_on="s_nationkey", right_on="n_nationkey")\n'
            '    .filter(pl.col("n_name") == "ALGERIA")\n'
            '    .select("ps_partkey")\n'
            ')\n'
            'result = (\n'
            '    part\n'
            '    .join(algerian_parts, left_on="p_partkey", right_on="ps_partkey", how="semi")\n'
            '    .select(["p_partkey", "p_name"])\n'
            '    .sort("p_partkey")\n'
            '    .head(20)\n'
            ')'
        ),
    },
]
