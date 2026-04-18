"""Extended grammar-constrained tests using test_gram.build_gbnf + llama_cpp.

Loads the model once, then runs multiple test cases with different schemas
and tasks. Each result is validated against the project's Lark grammar.

Run:
    python -m test_gram.test_more
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.polars_grammar import validate as lark_validate  # noqa: E402
from test_gram.test_gram import DEFAULT_FILE, DEFAULT_REPO, build_gbnf  # noqa: E402

# ---------------------------------------------------------------------------
# Test cases: (name, schema, task)
# ---------------------------------------------------------------------------

CASES = [
    (
        "filter + select",
        {
            "customer": {"columns": {
                "c_custkey": "Int64",
                "c_name": "Utf8",
                "c_mktsegment": "Utf8",
                "c_acctbal": "Float64",
            }}
        },
        'From `customer`, keep rows where c_mktsegment == "BUILDING", '
        "then select c_custkey and c_name.",
    ),
    (
        "sort + head",
        {
            "orders": {"columns": {
                "o_orderkey": "Int64",
                "o_custkey": "Int64",
                "o_totalprice": "Float64",
                "o_orderdate": "Date",
                "o_orderstatus": "Utf8",
            }}
        },
        "From `orders`, sort by o_totalprice descending and keep the top 5 rows.",
    ),
    (
        "group_by + agg",
        {
            "lineitem": {"columns": {
                "l_orderkey": "Int64",
                "l_returnflag": "Utf8",
                "l_quantity": "Float64",
                "l_extendedprice": "Float64",
                "l_discount": "Float64",
            }}
        },
        "From `lineitem`, group by l_returnflag and compute the sum of "
        "l_quantity and the mean of l_discount.",
    ),
    (
        "with_columns + alias",
        {
            "lineitem": {"columns": {
                "l_orderkey": "Int64",
                "l_extendedprice": "Float64",
                "l_discount": "Float64",
                "l_tax": "Float64",
            }}
        },
        "From `lineitem`, add a column `revenue` equal to "
        "l_extendedprice * (1 - l_discount).",
    ),
    (
        "filter numeric",
        {
            "customer": {"columns": {
                "c_custkey": "Int64",
                "c_name": "Utf8",
                "c_acctbal": "Float64",
            }}
        },
        "From `customer`, keep rows where c_acctbal > 1000, then sort by "
        "c_acctbal descending.",
    ),
    (
        "unique",
        {
            "orders": {"columns": {
                "o_orderkey": "Int64",
                "o_custkey": "Int64",
                "o_orderstatus": "Utf8",
            }}
        },
        "From `orders`, return unique values of o_orderstatus.",
    ),
]


def run() -> int:
    try:
        from llama_cpp import Llama, LlamaGrammar
    except ImportError:
        print("llama-cpp-python not installed.")
        return 1

    print(f"Loading model {DEFAULT_FILE}...")
    llm = Llama.from_pretrained(
        repo_id=DEFAULT_REPO,
        filename=DEFAULT_FILE,
        n_ctx=131072,
        n_gpu_layers=-1,
        verbose=False,
    )

    system = (
        "Return only valid Python Polars code (no markdown, no comments). "
        "Assign the final DataFrame to `result`. "
        "Use only the tables and columns from the schema provided."
    )

    passed = 0
    failed = 0

    for name, schema, task in CASES:
        import json
        prompt = (
            f"Schema: {json.dumps(schema)}\n"
            f"Task: {task}"
        )
        gbnf = build_gbnf(schema)
        grammar = LlamaGrammar.from_string(gbnf)

        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            grammar=grammar,
            max_tokens=256,
            temperature=0.0,
        )
        text = out["choices"][0]["message"]["content"].strip()
        ok, err = lark_validate(text, schema)

        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"       lark: {err}")
            print(f"       code: {text}")
            failed += 1
        else:
            passed += 1

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(run())
