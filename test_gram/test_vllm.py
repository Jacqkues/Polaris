"""Grammar-constrained Polars generation using pygbnf + vLLM (gemma-4-E2B-it).

vLLM applies the GBNF grammar via xgrammar during inference — same grammar as
test_gram.py but without llama_cpp. Faster on GPU thanks to vLLM's continuous
batching and PagedAttention.

Requires:
    pip install vllm pygbnf

Run:
    python -m test_gram.test_vllm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.polars_grammar import validate as lark_validate  # noqa: E402
from test_gram.test_gram import build_gbnf  # noqa: E402

MODEL = "mistralai/Ministral-3B-Instruct-2410"

SYSTEM = (
    "Return only valid Python Polars code (no markdown, no comments). "
    "Assign the final DataFrame to `result`. "
    "Use only the tables and columns from the schema provided."
)

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
   

    print(f"Loading {MODEL} with vLLM...")
    llm = LLM(model=MODEL, dtype="float16")

    passed = 0
    failed = 0

    for name, schema, task in CASES:
        prompt = f"Schema: {json.dumps(schema)}\nTask: {task}"
        gbnf = build_gbnf(schema)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            structured_outputs=StructuredOutputsParams(grammar=gbnf),
        )

        outputs = llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            sampling_params=sampling_params,
        )
        text = outputs[0].outputs[0].text.strip()
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
