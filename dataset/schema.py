from pydantic import BaseModel, Field


class TableSchema(BaseModel):
    columns: dict[str, str]
    n_rows: int | None = None


class DatasetRecord(BaseModel):
    id: str
    tables: dict[str, TableSchema]
    question: str
    reference_code: str
    tags: list[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5, default=3)
    expected_output_hash: str | None = None
    expected_n_rows: int | None = None
    expected_columns: list[str] | None = None


TPCH_TABLES: dict[str, dict[str, str]] = {
    "region": {
        "r_regionkey": "Int64",
        "r_name": "Utf8",
        "r_comment": "Utf8",
    },
    "nation": {
        "n_nationkey": "Int64",
        "n_name": "Utf8",
        "n_regionkey": "Int64",
        "n_comment": "Utf8",
    },
    "part": {
        "p_partkey": "Int64",
        "p_name": "Utf8",
        "p_mfgr": "Utf8",
        "p_brand": "Utf8",
        "p_type": "Utf8",
        "p_size": "Int64",
        "p_container": "Utf8",
        "p_retailprice": "Float64",
        "p_comment": "Utf8",
    },
    "supplier": {
        "s_suppkey": "Int64",
        "s_name": "Utf8",
        "s_address": "Utf8",
        "s_nationkey": "Int64",
        "s_phone": "Utf8",
        "s_acctbal": "Float64",
        "s_comment": "Utf8",
    },
    "partsupp": {
        "ps_partkey": "Int64",
        "ps_suppkey": "Int64",
        "ps_availqty": "Int64",
        "ps_supplycost": "Float64",
        "ps_comment": "Utf8",
    },
    "customer": {
        "c_custkey": "Int64",
        "c_name": "Utf8",
        "c_address": "Utf8",
        "c_nationkey": "Int64",
        "c_phone": "Utf8",
        "c_acctbal": "Float64",
        "c_mktsegment": "Utf8",
        "c_comment": "Utf8",
    },
    "orders": {
        "o_orderkey": "Int64",
        "o_custkey": "Int64",
        "o_orderstatus": "Utf8",
        "o_totalprice": "Float64",
        "o_orderdate": "Date",
        "o_orderpriority": "Utf8",
        "o_clerk": "Utf8",
        "o_shippriority": "Int64",
        "o_comment": "Utf8",
    },
    "lineitem": {
        "l_orderkey": "Int64",
        "l_partkey": "Int64",
        "l_suppkey": "Int64",
        "l_linenumber": "Int64",
        "l_quantity": "Float64",
        "l_extendedprice": "Float64",
        "l_discount": "Float64",
        "l_tax": "Float64",
        "l_returnflag": "Utf8",
        "l_linestatus": "Utf8",
        "l_shipdate": "Date",
        "l_commitdate": "Date",
        "l_receiptdate": "Date",
        "l_shipinstruct": "Utf8",
        "l_shipmode": "Utf8",
        "l_comment": "Utf8",
    },
}
