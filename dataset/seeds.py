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
    {
        "id": "tpch_seed_011_bool_or_filter",
        "tags": ["filter", "bool_or", "groupby"],
        "difficulty": 2,
        "tables_used": ["orders"],
        "question": "Count orders per status, restricted to status 'F' or 'O'.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .filter((pl.col("o_orderstatus") == "F") | (pl.col("o_orderstatus") == "O"))\n'
            '    .group_by("o_orderstatus")\n'
            '    .agg(pl.len().alias("n"))\n'
            '    .sort("o_orderstatus")\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_012_not_filter",
        "tags": ["filter", "not"],
        "difficulty": 2,
        "tables_used": ["customer"],
        "question": "Return the first 5 customers whose market segment is not BUILDING.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .filter(~(pl.col("c_mktsegment") == "BUILDING"))\n'
            '    .select(["c_custkey", "c_mktsegment"])\n'
            '    .sort("c_custkey")\n'
            '    .head(5)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_013_n_unique",
        "tags": ["agg", "n_unique"],
        "difficulty": 1,
        "tables_used": ["orders"],
        "question": "How many distinct customers have placed at least one order?",
        "reference_code": (
            'result = orders.select(pl.col("o_custkey").n_unique().alias("distinct_customers"))'
        ),
    },
    {
        "id": "tpch_seed_014_cast",
        "tags": ["cast", "with_columns"],
        "difficulty": 2,
        "tables_used": ["part"],
        "question": "Return the first 5 parts with p_size cast to Float64.",
        "reference_code": (
            'result = (\n'
            '    part\n'
            '    .with_columns(pl.col("p_size").cast(pl.Float64).alias("p_size_f"))\n'
            '    .select(["p_partkey", "p_size_f"])\n'
            '    .sort("p_partkey")\n'
            '    .head(5)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_015_str_starts_with",
        "tags": ["filter", "string"],
        "difficulty": 2,
        "tables_used": ["customer"],
        "question": "First 10 customers whose phone number starts with '11'.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .filter(pl.col("c_phone").str.starts_with("11"))\n'
            '    .select(["c_custkey", "c_phone"])\n'
            '    .sort("c_custkey")\n'
            '    .head(10)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_016_dt_year_groupby",
        "tags": ["date", "dt_year", "groupby", "agg"],
        "difficulty": 3,
        "tables_used": ["orders"],
        "question": "Number of orders per year.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .with_columns(pl.col("o_orderdate").dt.year().alias("year"))\n'
            '    .group_by("year")\n'
            '    .agg(pl.len().alias("n_orders"))\n'
            '    .sort("year")\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_017_multi_agg_list",
        "tags": ["groupby", "agg_list"],
        "difficulty": 3,
        "tables_used": ["lineitem"],
        "question": "Per return flag: sum of quantity, mean extended price, max discount.",
        "reference_code": (
            'result = (\n'
            '    lineitem\n'
            '    .group_by("l_returnflag")\n'
            '    .agg([\n'
            '        pl.col("l_quantity").sum().alias("sum_qty"),\n'
            '        pl.col("l_extendedprice").mean().alias("avg_price"),\n'
            '        pl.col("l_discount").max().alias("max_disc"),\n'
            '    ])\n'
            '    .sort("l_returnflag")\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_018_groupby_multi",
        "tags": ["groupby_multi", "agg"],
        "difficulty": 2,
        "tables_used": ["orders"],
        "question": "Count orders grouped by status and priority.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .group_by(["o_orderstatus", "o_orderpriority"])\n'
            '    .agg(pl.len().alias("n"))\n'
            '    .sort(["o_orderstatus", "o_orderpriority"])\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_019_with_columns_multi",
        "tags": ["with_columns_list", "string", "arith"],
        "difficulty": 3,
        "tables_used": ["customer"],
        "question": "Return 5 customers with upper-cased name and doubled account balance.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .with_columns([\n'
            '        pl.col("c_name").str.to_uppercase().alias("c_name_upper"),\n'
            '        (pl.col("c_acctbal") * 2).alias("double_bal"),\n'
            '    ])\n'
            '    .select(["c_custkey", "c_name_upper", "double_bal"])\n'
            '    .sort("c_custkey")\n'
            '    .head(5)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_020_rename",
        "tags": ["rename", "select"],
        "difficulty": 1,
        "tables_used": ["customer"],
        "question": "Rename c_custkey to id and c_name to name; return first 5 rows sorted by id.",
        "reference_code": (
            'result = (\n'
            '    customer\n'
            '    .select(["c_custkey", "c_name"])\n'
            '    .rename({"c_custkey": "id", "c_name": "name"})\n'
            '    .sort("id")\n'
            '    .head(5)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_021_unique_noargs",
        "tags": ["unique"],
        "difficulty": 1,
        "tables_used": ["orders"],
        "question": "Return the distinct (status, priority) pairs sorted lexicographically.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .select(["o_orderstatus", "o_orderpriority"])\n'
            '    .unique()\n'
            '    .sort(["o_orderstatus", "o_orderpriority"])\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_022_left_join",
        "tags": ["join_left", "select"],
        "difficulty": 2,
        "tables_used": ["nation", "region"],
        "question": "First 5 nations with their region name, left-joined on region key.",
        "reference_code": (
            'result = (\n'
            '    nation\n'
            '    .join(region, left_on="n_regionkey", right_on="r_regionkey", how="left")\n'
            '    .select([pl.col("n_name"), pl.col("r_name").alias("region")])\n'
            '    .sort("n_name")\n'
            '    .head(5)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_023_three_way_join_arith",
        "tags": ["join", "filter", "arith", "sort", "limit"],
        "difficulty": 4,
        "tables_used": ["lineitem", "orders", "customer"],
        "question": "Top 10 lineitems (by price * (1-disc) * (1+tax)) for AUTOMOBILE customers.",
        "reference_code": (
            'result = (\n'
            '    lineitem\n'
            '    .join(orders, left_on="l_orderkey", right_on="o_orderkey")\n'
            '    .join(customer, left_on="o_custkey", right_on="c_custkey")\n'
            '    .filter(pl.col("c_mktsegment") == "AUTOMOBILE")\n'
            '    .with_columns(\n'
            '        (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) * (1 + pl.col("l_tax")))\n'
            '        .alias("gross")\n'
            '    )\n'
            '    .select(["c_name", "l_orderkey", "gross"])\n'
            '    .sort("gross", descending=True)\n'
            '    .head(10)\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_024_bool_and_date",
        "tags": ["filter", "bool_and", "date"],
        "difficulty": 3,
        "tables_used": ["lineitem"],
        "question": "Count lineitems shipped in 1994 with quantity > 30 and discount < 0.05.",
        "reference_code": (
            'result = (\n'
            '    lineitem\n'
            '    .filter(\n'
            '        (pl.col("l_shipdate").is_between(pl.date(1994, 1, 1), pl.date(1994, 12, 31)))\n'
            '        & (pl.col("l_quantity") > 30)\n'
            '        & (pl.col("l_discount") < 0.05)\n'
            '    )\n'
            '    .select(pl.len().alias("n"))\n'
            ')'
        ),
    },
    {
        "id": "tpch_seed_025_dt_month_filter",
        "tags": ["date", "dt_month", "filter", "groupby"],
        "difficulty": 3,
        "tables_used": ["orders"],
        "question": "Count March orders across all years.",
        "reference_code": (
            'result = (\n'
            '    orders\n'
            '    .with_columns(pl.col("o_orderdate").dt.month().alias("month"))\n'
            '    .filter(pl.col("month") == 3)\n'
            '    .group_by("month")\n'
            '    .agg(pl.len().alias("n_march"))\n'
            ')'
        ),
    },
]
