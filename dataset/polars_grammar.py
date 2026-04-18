"""Build a Lark grammar for Polars code with schema-restricted table/column names.

The grammar constrains an LLM (via Outlines or similar) so that:
  - only provided table names appear as chain roots / join targets
  - only provided column names appear inside `pl.col(...)` and in column-ref
    positions like group_by/sort/over/join keys
  - the overall code structure is a valid subset of Polars method chains

Arbitrary strings used as literals (e.g. `.filter(pl.col("x") == "BUILDING")`)
are still allowed via the generic STRING terminal.

Entry shape:
    result = (
        <table>
        .op(...)
        ...
    )

Covered ops:
    filter, select, with_columns, group_by, agg, join, sort, head, limit,
    unique, rename
Covered expression methods:
    .alias, .sum/mean/min/max/count/n_unique/first/last, .is_between,
    .rank(method=..., descending=...), .over, .str.contains/starts_with/
    ends_with/to_lowercase/to_uppercase, .dt.year/month/day, .cast
Covered atoms:
    pl.col(COL), pl.len(), pl.date(Y,M,D), STRING, INT, FLOAT, BOOL, (expr)
Operators:
    ==, !=, <, <=, >, >=, +, -, *, /, &, |, ~
"""
from __future__ import annotations

from pathlib import Path

BASE_GRAMMAR = r"""
?start: assign

assign: "result" "=" rhs

rhs: "(" pipeline ")" | pipeline

pipeline: table_ref (method)+

table_ref: TABLENAME

method: "." meth_call

?meth_call: filter_call
          | select_call
          | with_cols_call
          | group_by_call
          | agg_call
          | join_call
          | sort_call
          | head_call
          | limit_call
          | unique_call
          | rename_call

filter_call: "filter" "(" expr ")"

select_call: "select" "(" expr_or_list ")"

with_cols_call: "with_columns" "(" expr_or_list ")"

agg_call: "agg" "(" expr_or_list ")"

group_by_call: "group_by" "(" colref_or_list ")"

sort_call: "sort" "(" sort_key ("," sort_kw)? ")"
sort_key: colref | colref_list
sort_kw: "descending" "=" (BOOL | bool_list)

head_call: "head" "(" INT ")"
limit_call: "limit" "(" INT ")"

unique_call: "unique" "(" (colref_or_list)? ")"

rename_call: "rename" "(" "{" rename_pair ("," rename_pair)* "}" ")"
rename_pair: colref ":" STRING

join_call: "join" "(" TABLENAME ("," join_kw)+ ")"
join_kw: "left_on" "=" colref
       | "right_on" "=" colref
       | "on" "=" colref_or_list
       | "how" "=" STRING

// colref = any quoted name in a column-reference position. Prefers COLSTR
// (schema-declared) via terminal priority, but falls back to STRING so that
// alias names created mid-chain (e.g. .alias("revenue")) can be referenced
// later in the same expression.
colref: COLSTR | STRING
colref_or_list: colref | colref_list
// Trailing commas are allowed throughout, matching Python's list literal rules.
colref_list: "[" colref ("," colref)* ","? "]"
colstr_list: "[" COLSTR ("," COLSTR)* ","? "]"
bool_list: "[" BOOL ("," BOOL)* ","? "]"
expr_or_list: expr | expr_list
expr_list: "[" expr ("," expr)* ","? "]"

?expr: or_expr
?or_expr: and_expr ("|" and_expr)*
?and_expr: not_expr ("&" not_expr)*
?not_expr: "~" not_expr | cmp_expr
?cmp_expr: sum_expr (cmp_op sum_expr)?
!cmp_op: "==" | "!=" | "<=" | ">=" | "<" | ">"
?sum_expr: prod_expr (sum_op prod_expr)*
!sum_op: "+" | "-"
?prod_expr: unary (prod_op unary)*
!prod_op: "*" | "/"
?unary: "-" atom_with_methods | atom_with_methods

atom_with_methods: expr_atom (expr_method)* | literal_atom

?expr_atom: col_atom | pl_len | pl_date | paren_expr
?literal_atom: FLOAT | INT | STRING | COLSTR | BOOL

paren_expr: "(" expr ")"
col_atom: "pl.col" "(" colref ")"
pl_len: "pl.len" "(" ")"
pl_date: "pl.date" "(" INT "," INT "," INT ")"

expr_method: "." mcall
?mcall: m_alias | m_agg | m_is_between | m_rank | m_over | m_cast | m_nsstr | m_nsdt
m_alias: "alias" "(" STRING ")"
m_agg: AGGNAME "(" ")"
AGGNAME: "sum" | "mean" | "min" | "max" | "count" | "n_unique" | "first" | "last" | "len"
m_is_between: "is_between" "(" expr "," expr ")"
m_rank: "rank" "(" (rank_kw ("," rank_kw)*)? ")"
rank_kw: "method" "=" STRING
       | "descending" "=" BOOL
m_over: "over" "(" colref_or_list ")"
m_cast: "cast" "(" pl_type ")"
pl_type: "pl.Int64" | "pl.Int32" | "pl.Float64" | "pl.Float32" | "pl.Utf8" | "pl.String" | "pl.Date" | "pl.Datetime" | "pl.Boolean"

m_nsstr: "str" "." strmeth
?strmeth: strm_contains | strm_sw | strm_ew | strm_lower | strm_upper | strm_len
strm_contains: "contains" "(" STRING ")"
strm_sw: "starts_with" "(" STRING ")"
strm_ew: "ends_with" "(" STRING ")"
strm_lower: "to_lowercase" "(" ")"
strm_upper: "to_uppercase" "(" ")"
strm_len: "len_chars" "(" ")"

m_nsdt: "dt" "." dtmeth
?dtmeth: dtm_year | dtm_month | dtm_day
dtm_year: "year" "(" ")"
dtm_month: "month" "(" ")"
dtm_day: "day" "(" ")"

BOOL: "True" | "False"
INT: /[0-9]+/
FLOAT: /[0-9]+\.[0-9]+/

%TABLENAME_DECL%
%COLSTR_DECL%
STRING.1: /"[^"\\]*(?:\\.[^"\\]*)*"/

%import common.WS
%ignore WS
"""


def _escape_for_lark_string_terminal(s: str) -> str:
    """Escape a column/table name so it can go inside a Lark string terminal.

    Lark string terminals use backslash-escaping for embedded double quotes,
    same as Python. Column/table names are normally plain identifiers, so this
    is defensive.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _table_terminal(tables: list[str]) -> str:
    alts = " | ".join(f'"{_escape_for_lark_string_terminal(t)}"' for t in tables)
    return f"TABLENAME.2: {alts}"


def _col_terminal(columns: list[str]) -> str:
    # Each alternative matches the quoted form, e.g. "c_name" in the source.
    alts = " | ".join(
        f'"\\"{_escape_for_lark_string_terminal(c)}\\""' for c in columns
    )
    return f"COLSTR.2: {alts}"


def _extract_schema(tables: dict) -> tuple[list[str], list[str]]:
    """Return `(sorted_table_names, sorted_column_names)` from a tables dict.

    Accepts the shapes used across the project:
        {"customer": {"columns": {"c_name": "Utf8", ...}, "n_rows": 1500}}
        {"customer": {"c_name": "Utf8", ...}}
        {"customer": ["c_name", "c_custkey"]}
    """
    if not tables:
        raise ValueError("tables must be non-empty to build a grammar")

    table_names: list[str] = []
    all_cols: set[str] = set()

    for t, meta in tables.items():
        table_names.append(t)
        if isinstance(meta, dict):
            cols = meta.get("columns", meta)
        else:
            cols = meta
        if isinstance(cols, dict):
            all_cols.update(cols.keys())
        else:
            all_cols.update(cols)

    if not all_cols:
        raise ValueError("no columns found across provided tables")

    return sorted(set(table_names)), sorted(all_cols)


def build_grammar(tables: dict) -> str:
    """Return a Lark grammar string specialized for the given table schemas.

    Used by the local self-test (Lark parses faster than any GBNF runtime and
    gives clean diagnostics). For constrained decoding at inference time via
    xgrammar, use `build_grammar_gbnf` instead.
    """
    table_names, col_list = _extract_schema(tables)
    grammar = BASE_GRAMMAR.replace("%TABLENAME_DECL%", _table_terminal(table_names))
    grammar = grammar.replace("%COLSTR_DECL%", _col_terminal(col_list))
    return grammar


# GBNF (llama.cpp-style EBNF) grammar — the format xgrammar and other
# structured-generation runtimes expect. Semantically equivalent to
# BASE_GRAMMAR above but expressed without Lark-specific features:
#   - `::=` instead of `:` for rule definition
#   - whitespace is NOT auto-ignored; we thread `ws` between tokens
#   - no terminal priorities (ambiguity between string and colstr is
#     resolved by the parser exploring both branches — both are valid
#     parses so correctness is preserved)
BASE_GRAMMAR_GBNF = r"""
root ::= assign
assign ::= "result" ws "=" ws rhs
rhs ::= "(" ws pipeline ws ")" | pipeline
pipeline ::= tableref (ws method)+
tableref ::= tablename

method ::= "." ws methcall

methcall ::= filtercall | selectcall | withcolscall | groupbycall
           | aggcall | joincall | sortcall | headcall | limitcall
           | uniquecall | renamecall

filtercall ::= "filter" ws "(" ws expr ws ")"
selectcall ::= "select" ws "(" ws exprorlist ws ")"
withcolscall ::= "with_columns" ws "(" ws exprorlist ws ")"
aggcall ::= "agg" ws "(" ws exprorlist ws ")"

groupbycall ::= "group_by" ws "(" ws colreforlist ws ")"

sortcall ::= "sort" ws "(" ws sortkey (ws "," ws sortkw)? ws ")"
sortkey ::= colref | colreflist
sortkw ::= "descending" ws "=" ws (bool | boollist)

headcall ::= "head" ws "(" ws int ws ")"
limitcall ::= "limit" ws "(" ws int ws ")"

uniquecall ::= "unique" ws "(" ws (colreforlist)? ws ")"

renamecall ::= "rename" ws "(" ws "{" ws renamepair (ws "," ws renamepair)* (ws ",")? ws "}" ws ")"
renamepair ::= colref ws ":" ws string

joincall ::= "join" ws "(" ws tablename (ws "," ws joinkw)+ ws ")"
joinkw ::= "left_on" ws "=" ws colref
         | "right_on" ws "=" ws colref
         | "on" ws "=" ws colreforlist
         | "how" ws "=" ws string

colref ::= colstr | string
colreforlist ::= colref | colreflist
colreflist ::= "[" ws colref (ws "," ws colref)* (ws ",")? ws "]"
boollist ::= "[" ws bool (ws "," ws bool)* (ws ",")? ws "]"
exprorlist ::= expr | exprlist
exprlist ::= "[" ws expr (ws "," ws expr)* (ws ",")? ws "]"

expr ::= orexpr
orexpr ::= andexpr (ws "|" ws andexpr)*
andexpr ::= notexpr (ws "&" ws notexpr)*
notexpr ::= "~" ws notexpr | cmpexpr
cmpexpr ::= sumexpr (ws cmpop ws sumexpr)?
cmpop ::= "==" | "!=" | "<=" | ">=" | "<" | ">"
sumexpr ::= prodexpr (ws sumop ws prodexpr)*
sumop ::= "+" | "-"
prodexpr ::= unary (ws prodop ws unary)*
prodop ::= "*" | "/"
unary ::= "-" ws atomwithmethods | atomwithmethods

atomwithmethods ::= expratom (exprmethod)* | literalatom

expratom ::= colatom | pllen | pldate | parenexpr
literalatom ::= float | int | string | colstr | bool

parenexpr ::= "(" ws expr ws ")"
colatom ::= "pl.col" ws "(" ws colref ws ")"
pllen ::= "pl.len" ws "(" ws ")"
pldate ::= "pl.date" ws "(" ws int ws "," ws int ws "," ws int ws ")"

exprmethod ::= "." ws mcall
mcall ::= malias | magg | misbetween | mrank | mover | mcast | mnsstr | mnsdt
malias ::= "alias" ws "(" ws string ws ")"
magg ::= aggname ws "(" ws ")"
aggname ::= "sum" | "mean" | "min" | "max" | "count" | "n_unique" | "first" | "last" | "len"
misbetween ::= "is_between" ws "(" ws expr ws "," ws expr ws ")"
mrank ::= "rank" ws "(" ws (rankkw (ws "," ws rankkw)*)? ws ")"
rankkw ::= "method" ws "=" ws string | "descending" ws "=" ws bool
mover ::= "over" ws "(" ws colreforlist ws ")"
mcast ::= "cast" ws "(" ws pltype ws ")"
pltype ::= "pl.Int64" | "pl.Int32" | "pl.Float64" | "pl.Float32" | "pl.Utf8" | "pl.String" | "pl.Date" | "pl.Datetime" | "pl.Boolean"

mnsstr ::= "str" ws "." ws strmeth
strmeth ::= strmcontains | strmsw | strmew | strmlower | strmupper | strmlen
strmcontains ::= "contains" ws "(" ws string ws ")"
strmsw ::= "starts_with" ws "(" ws string ws ")"
strmew ::= "ends_with" ws "(" ws string ws ")"
strmlower ::= "to_lowercase" ws "(" ws ")"
strmupper ::= "to_uppercase" ws "(" ws ")"
strmlen ::= "len_chars" ws "(" ws ")"

mnsdt ::= "dt" ws "." ws dtmeth
dtmeth ::= "year" ws "(" ws ")" | "month" ws "(" ws ")" | "day" ws "(" ws ")"

bool ::= "True" | "False"
int ::= [0-9]+
float ::= [0-9]+ "." [0-9]+
string ::= "\"" ([^"\\] | "\\" [\x00-\x7f])* "\""
ws ::= [ \t\n\r]*

%TABLENAME_GBNF%
%COLSTR_GBNF%
"""


def _gbnf_escape(s: str) -> str:
    """Escape a literal string so it can be embedded in a GBNF `"..."` terminal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _table_gbnf(tables: list[str]) -> str:
    alts = " | ".join(f'"{_gbnf_escape(t)}"' for t in tables)
    return f"tablename ::= {alts}"


def _col_gbnf(columns: list[str]) -> str:
    # Each alternative matches the quoted form as seen in source, e.g. "c_name".
    alts = " | ".join(f'"\\"{_gbnf_escape(c)}\\""' for c in columns)
    return f"colstr ::= {alts}"


def build_grammar_gbnf(tables: dict) -> str:
    """Return a GBNF grammar string (for xgrammar / llama.cpp-style engines)
    specialized for the given table schemas."""
    table_names, col_list = _extract_schema(tables)
    grammar = BASE_GRAMMAR_GBNF.replace("%TABLENAME_GBNF%", _table_gbnf(table_names))
    grammar = grammar.replace("%COLSTR_GBNF%", _col_gbnf(col_list))
    return grammar


def validate(code: str, tables: dict) -> tuple[bool, str | None]:
    """Parse `code` against the grammar built from `tables`. Returns (ok, err)."""
    from lark import Lark, LarkError

    grammar = build_grammar(tables)
    try:
        parser = Lark(grammar, parser="lalr")
    except LarkError as e:
        return False, f"grammar-compile-error: {e}"
    try:
        parser.parse(code)
        return True, None
    except LarkError as e:
        return False, str(e)


def _self_test() -> int:
    """Run the grammar against every reference_code in data/seeds.jsonl."""
    import json

    seeds_path = Path(__file__).resolve().parent.parent / "data" / "seeds.jsonl"
    if not seeds_path.exists():
        print(f"seeds file not found: {seeds_path}")
        return 1

    ok = 0
    fail = 0
    first_failures: list[tuple[str, str]] = []
    with seeds_path.open() as f:
        for line in f:
            rec = json.loads(line)
            code = rec["reference_code"]
            tables = rec["tables"]
            # Seeds have multi-statement oracles; grammar handles only the
            # final `result = ...` assignment. Extract it if present.
            target = _extract_result_block(code)
            okp, err = validate(target, tables)
            if okp:
                ok += 1
                print(f"  OK   {rec['id']}")
            else:
                fail += 1
                print(f"  FAIL {rec['id']}  {err.splitlines()[0] if err else ''}")
                if len(first_failures) < 3:
                    first_failures.append((rec["id"], err or ""))

    print(f"\n{ok} / {ok + fail} seeds parse cleanly")
    if first_failures:
        print("\nFirst failures (detail):")
        for sid, err in first_failures:
            print(f"\n--- {sid} ---\n{err}")

    # Hand-crafted POSITIVE cases that exercise grammar features not covered
    # by the execution-tested seeds.
    print("\n-- extra positive cases (should PASS) --")
    tpch_tables = {
        "customer": {"columns": {
            "c_custkey": "Int64", "c_name": "Utf8", "c_address": "Utf8",
            "c_nationkey": "Int64", "c_phone": "Utf8", "c_acctbal": "Float64",
            "c_mktsegment": "Utf8", "c_comment": "Utf8",
        }},
        "orders": {"columns": {
            "o_orderkey": "Int64", "o_custkey": "Int64", "o_orderstatus": "Utf8",
            "o_totalprice": "Float64", "o_orderdate": "Date",
            "o_orderpriority": "Utf8", "o_clerk": "Utf8",
            "o_shippriority": "Int64", "o_comment": "Utf8",
        }},
        "lineitem": {"columns": {
            "l_orderkey": "Int64", "l_partkey": "Int64", "l_suppkey": "Int64",
            "l_quantity": "Float64", "l_extendedprice": "Float64",
            "l_discount": "Float64", "l_tax": "Float64",
            "l_returnflag": "Utf8", "l_shipdate": "Date",
        }},
    }
    pos_extra = [
        ("str.ends_with", 'result = customer.filter(pl.col("c_name").str.ends_with("01"))'),
        ("str.to_lowercase", 'result = customer.with_columns(pl.col("c_name").str.to_lowercase().alias("lo"))'),
        ("str.len_chars", 'result = customer.with_columns(pl.col("c_name").str.len_chars().alias("n"))'),
        ("dt.day", 'result = orders.with_columns(pl.col("o_orderdate").dt.day().alias("d"))'),
        ("cast_int64", 'result = orders.with_columns(pl.col("o_totalprice").cast(pl.Int64).alias("i"))'),
        ("cast_utf8", 'result = orders.with_columns(pl.col("o_orderkey").cast(pl.Utf8).alias("s"))'),
        (
            "over_multi_partition",
            'result = orders.with_columns('
            'pl.col("o_totalprice").rank(method="dense").over(["o_orderstatus", "o_orderpriority"]).alias("r"))',
        ),
        (
            "agg_list_mixed",
            'result = lineitem.group_by("l_returnflag").agg([pl.col("l_quantity").sum(), pl.col("l_discount").mean().alias("avg_disc"), pl.len().alias("n")])',
        ),
        ("no_outer_parens", 'result = orders.head(5)'),
        ("chained_filter", 'result = orders.filter(pl.col("o_totalprice") > 1000).filter(pl.col("o_orderstatus") == "F").head(5)'),
        (
            "select_mixed_expr_and_col",
            'result = customer.select([pl.col("c_name"), (pl.col("c_acctbal") + 1).alias("plus_one")]).head(3)',
        ),
        ("neg_literal", 'result = customer.filter(pl.col("c_acctbal") > -100).head(5)'),
        ("bool_literal_false", 'result = orders.sort("o_totalprice", descending=False).head(3)'),
    ]
    pos_extra_ok = 0
    for name, code in pos_extra:
        okp, err = validate(code, tpch_tables)
        tag = "OK" if okp else "FAIL"
        print(f"  {tag}  {name}" + ("" if okp else f"  :: {(err or '').splitlines()[0]}"))
        if okp:
            pos_extra_ok += 1
    print(f"\n{pos_extra_ok} / {len(pos_extra)} extra positive cases parsed")

    # Negative checks. Note the intentional scope: the grammar strictly
    # constrains *table names* and *overall structure*. Column-ref positions
    # accept any quoted string so that mid-chain aliases (.alias("revenue")
    # then .sort("revenue")) remain expressible — bogus columns are caught at
    # execution time via the existing compare/oracle path, not here.
    print("\n-- negative checks (should FAIL) --")
    neg_tables = {"customer": {"columns": {"c_name": "Utf8", "c_custkey": "Int64"}}}
    neg_cases = [
        ("bogus-table", 'result = fake_table.select(["c_name"])'),
        ("bare-python-not-polars", 'result = [1, 2, 3]'),
        ("missing-result-assign", 'customer.select(["c_name"])'),
        ("bogus-method", 'result = customer.zap("c_name")'),
        ("bogus-agg", 'result = customer.group_by("c_name").agg(pl.col("c_custkey").medianX())'),
        ("bogus-str-method", 'result = customer.filter(pl.col("c_name").str.weirdcall("x"))'),
        ("bogus-dt-method", 'result = customer.filter(pl.col("c_name").dt.century() == 2)'),
        ("bogus-cast-type", 'result = customer.with_columns(pl.col("c_acctbal").cast(pl.Decimal).alias("d"))'),
        ("bogus-assign-name", 'foo = customer.head(5)'),
        ("bad-operator", 'result = customer.filter(pl.col("c_acctbal") === 1)'),
        ("bad-walrus-inside", 'result = customer.filter(pl.col("c_acctbal") := 1)'),
        ("missing-closing-paren", 'result = customer.filter(pl.col("c_name") == "X"'),
        ("unquoted-col-in-pl.col", 'result = customer.filter(pl.col(c_name) == "X")'),
        ("chain-with-no-method", 'result = customer'),
        ("raw-python-expr", 'result = sum([1,2,3])'),
        ("distinct-not-polars", 'result = customer.distinct()'),
        ("string-literal-chained-over", 'result = customer.select("c_name".over("c_custkey"))'),
    ]
    neg_pass = 0
    for name, code in neg_cases:
        okp, _err = validate(code, neg_tables)
        tag = "REJECT-OK" if not okp else "ACCEPT-BUG"
        print(f"  {tag}  {name}")
        if not okp:
            neg_pass += 1
    print(f"\n{neg_pass} / {len(neg_cases)} negative cases correctly rejected")

    total_fail = (fail - 1) + (len(pos_extra) - pos_extra_ok) + (len(neg_cases) - neg_pass)
    return 0 if total_fail == 0 else 2


def _extract_result_block(code: str) -> str:
    """Return the `result = ...` statement (and its enclosing parens) from code.

    Seeds occasionally define helper variables before `result = ...`. We grab
    from the last `result =` to the matching close paren (or EOS).
    """
    idx = code.rfind("result =")
    if idx < 0:
        return code
    tail = code[idx:]
    # If this assigns a parenthesized chain, balance parens to find the end.
    eq_pos = tail.find("=")
    after_eq = tail[eq_pos + 1 :].lstrip()
    if after_eq.startswith("("):
        depth = 0
        for i, ch in enumerate(after_eq):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return "result = " + after_eq[: i + 1]
    return tail


if __name__ == "__main__":
    raise SystemExit(_self_test())
