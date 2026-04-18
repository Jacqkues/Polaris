# Polaris — Hackathon SLM Benchmarking

> Benchmarking Small Language Models on natural language → Polars code generation  
> Hackathon date: 2026-04-18 · Team: Polaris (3 people)

---

## The problem

Given a natural language question and a set of TPC-H table schemas, generate executable Python/Polars code that produces the correct `DataFrame`.

```
"Group the orders by customer and return the 10 with the highest total revenue."
  →  result = (
         orders
         .group_by("o_custkey")
         .agg(pl.col("o_totalprice").sum().alias("total_revenue"))
         .sort("total_revenue", descending=True)
         .head(10)
     )
```

The submission is scored as:

```
Score = N / (T × VRAM^0.1 × RAM^0.01)
```

where **N = correct outputs** and **T = total generation time**. Correctness dominates everything — getting +30% matches on a model that uses 3× more VRAM is still a net ×2.4 gain.

---

## What we built

### Evaluation harness

Rather than submitting blind and waiting for remote results, we built a local pipeline that reproduces the official metric exactly:

- **`dataset/gen_tpch.py`** — generates TPC-H tables locally
- **`dataset/executor.py`** — runs generated code in a restricted scope, captures `result`, handles exceptions
- **`dataset/compare.py`** — column-order-invariant DataFrame comparison with diagnostic reason codes
- **`data/seeds.jsonl`** — 25 validated seeds, each with question, schema, reference code, expected hash, tags, and difficulty
- **`benchmark_gemma.py`** — benchmarks Gemma 4 E2B with 4-tier scoring: `generated → parses → runs → matches`

### Models evaluated

| Model | Params | VRAM | matches |
|---|---|---|---|
| LiquidAI LFM2-8B-A1B | 8B (1B active, MoE) | ~16 GB | 10% |
| **Google Gemma 4 E2B-it** | 5B (2.3B effective) | ~10 GB | **30% → 84% runs** |

Gemma won by ×1.7 on the scoring formula despite being slower — correctness simply dominates.

### Prompt engineering progression

| Run | Additions | `runs` | `matches` |
|---|---|---|---|
| Baseline | Minimal prompt | 40% | 30% |
| v2 | Polars cheat sheet + 4 few-shot + typed schema | **84%** | 16%* |

*16% on a 2.5× harder seed set — not a regression. On shared seeds, v2 ≥ v0.

The cheat sheet explicitly forbids observed hallucinations: `.with_column` → `.with_columns`, `pl.desc(x)` → `.sort(x, descending=True)`, `df.len` → `df.height`, `.dense()` → `.rank(method="dense")`, boolean indexing → `.filter(...)`.

**Ceiling hit**: the remaining gap was not API errors but *aliasing conventions* — the model produces correct computations with different output column names (`total_quantity` vs the expected `sum_qty`). A prompt cannot fix a convention it has never seen.

---

## Constrained decoding — the core experiment

### Why prompts are not enough

A language model generates tokens one at a time with no hard guarantee about structure. Even with a detailed cheat sheet, the model can still:

- Use a deprecated API it saw during pretraining (`.dense()`, `df.len`)
- Reference a column that doesn't exist in the schema
- Generate syntactically invalid Python

Prompt instructions are *soft* — a constraint grammar is *hard*.

### Approach 1 — Dynamic Polars grammar (CFG)

`dataset/polars_grammar.py` builds a **per-request Lark grammar** that is instantiated with the actual table and column names for each question. This grammar covers:

- Only the tables provided in the schema as chain roots / join targets
- Only the columns provided as arguments to `pl.col(...)`, `group_by`, `sort`, `over`, `join`
- A structurally valid subset of Polars method chains: `filter`, `select`, `with_columns`, `group_by`, `agg`, `join`, `sort`, `head`, `limit`, `unique`, `rename`
- Expression methods: `.alias`, `.sum/.mean/.min/.max`, `.rank(method=...)`, `.over`, `.str.contains`, `.dt.year/.month/.day`, `.cast`

The grammar is fed to **Outlines** (with the `llguidance` backend) which compiles it into a logits processor. At each decoding step, the processor masks out any token whose addition would make the partial sequence unparseable by the grammar. The model can only emit structurally valid, schema-aware Polars code — **hallucinations like `pl.desc()` or unknown columns are physically impossible**.

```python
# gemma_model.py
from outlines.types import CFG
grammar = build_grammar(tables)           # per-request, injects real column names
response = outlines_model(prompt, CFG(grammar), max_new_tokens=512)
```

This is used in `GemmaModel.generate_constrained()` and wired into the three-level cascade in `gemma_cascade.py`:

```
L1 fast        → greedy, no constraint (~2-3s)
               → if static validation fails → L2
L2 constrained → Outlines CFG on Polars grammar (~5-8s)
               → if still fails → L3
L3 retry       → greedy with error feedback in the prompt
fallback       → best of L1/L2
```

Static validation (`looks_ok`) checks AST validity, presence of `result =`, known anti-patterns, and whether all `pl.col("x")` references exist in the schema — without executing the code.

### Approach 2 — JSON query plan with schema-constrained generation

`test_json_generator/` explores a different architecture: instead of generating Python directly, the model generates a **structured JSON query plan**, which a deterministic function then converts to Polars code.

```
question + schema
    → Gemma (JSON-constrained via Outlines)
    → QueryPlan (validated Pydantic object)
    → json_to_polars()
    → executable Polars code
```

The `QueryPlan` is a Pydantic model with a discriminated union of operations:

```python
class QueryPlan(BaseModel):
    source_table: str
    operations: list[
        Union[FilterOp, GroupByOp, SortOp, SelectOp, JoinOp, WithColumnsOp, HeadOp]
    ]
```

Outlines compiles the JSON schema of `QueryPlan` into a logits processor, so the model can only emit tokens that form valid JSON matching the schema. The result is then parsed with `QueryPlan.model_validate_json(result)`.

The converter `json_to_polars()` is a pure function with no model calls — it deterministically renders each operation into the correct modern Polars API:

```python
# FilterOp → .filter((pl.col("x") == "val") & (...))
# GroupByOp → .group_by(...).agg(pl.col(...).sum().alias(...))
# JoinOp    → .join(other, left_on="a", right_on="b", how="inner")
```

**Why this is interesting**: the JSON intermediate separates *what to compute* (model's job) from *how to write it in Polars* (converter's job). The model never needs to know the exact Polars API for rank, window functions, or date arithmetic — it only needs to express intent in a simple JSON vocabulary.

**Practical issues encountered**:

| Issue | Root cause | Fix |
|---|---|---|
| `outlines.generate` not found | Outlines v1 renamed to `outlines.Generator` | Use `outlines.Generator(model, QueryPlan)` |
| `max_tokens` kwarg rejected | Passes through to HF `model.generate` which uses `max_new_tokens` | Rename kwarg |
| JSON truncated mid-string | `discriminator` + `oneOf` schema confuses llguidance | Remove `Field(discriminator=...)`, use plain `Union` |
| `source_table: ":(orders)"` | Model confuses JSON format with SQL syntax | Add few-shot JSON examples in prompt |
| `on: "col1==col2"` in join | Model writes SQL join condition in wrong field | Split in renderer: detect `==`, emit `left_on`/`right_on` |
| `"o totalprice"` (space in column) | Constrained gen allows any string value | `_normalize_plan()` replaces space→underscore when result is a known column |
| `n: -1` in head | No integer constraint in schema | Clamp: `max(1, op.n)` in renderer |

The generation has a fallback: if the constrained pass produces invalid JSON (truncation, llguidance bug), it falls back to unconstrained greedy decoding and extracts the first `{...}` block from the output.

### Results on 5 seeds

After all fixes:

```
plan valid:   5/5  (100%)   JSON parsed by QueryPlan schema
converted:    5/5  (100%)   json_to_polars produced code
parses:       4/5   (80%)   AST-valid Python
runs:         1/5   (20%)   executes without error
matches:      0/5    (0%)   output matches reference
```

The remaining failures are model-quality issues (wrong column names, wrong alias names) that no amount of post-processing can fix without knowing the expected output. The JSON approach trades the Polars API hallucination problem for a semantic understanding problem.

---

## Repository structure

```
Polaris/
├── main.py                  # FastAPI server for submission
├── benchmark_gemma.py       # Benchmark harness — Gemma 4 E2B
├── gemma_model.py           # GemmaModel: greedy / constrained / retry
├── gemma_prompt.py          # System prompt, few-shot examples, schema formatter
├── gemma_cascade.py         # L1→L2→L3 cascade orchestrator + static validator
├── dataset/
│   ├── polars_grammar.py    # Dynamic Lark CFG — schema-aware Polars grammar
│   ├── executor.py          # Sandboxed code execution
│   ├── compare.py           # Column-order-invariant DataFrame comparison
│   └── gen_tpch.py          # TPC-H data generator
├── test_json_generator/
│   ├── json_query_generator.py  # QueryPlan schema + json_to_polars converter
│   └── bench_json.py            # Benchmark for the JSON generation approach
└── data/
    ├── seeds.jsonl          # 25 validated TPC-H seeds
    ├── tpch/                # TPC-H parquet tables
    └── expected/            # Reference output parquets
```

---

## Key takeaways

**What worked**
- Building a local evaluation harness before touching models — iterating in seconds instead of submitting blind.
- Reading the scoring formula carefully: `VRAM^0.1` is nearly flat, so quantization is a waste of time; correctness is everything.
- Prompt engineering *targeted at observed failures* — not generic rules. The cheat sheet went from a wishlist to a concrete list of bugs seen in the benchmark output.

**What didn't work as expected**
- The cheat sheet is not a guarantee. The model still hallucinates APIs it was explicitly told not to use.
- Aliasing conventions are not inferable from the question. This is the hard ceiling for prompt engineering.
- The `llguidance` dependency was absent on the submission VM — the CFG-constrained path, while implemented, couldn't be tested in time.

**The constrained decoding verdict**
- CFG grammar over generated Polars code: **eliminates structural and API hallucinations by construction**, but adds 2-3× latency and requires the `llguidance` backend.
- JSON plan with schema-constrained generation: **simpler schema, no Python grammar needed**, but pushes all semantic errors to the model's understanding of the JSON vocabulary. Interesting architecture, needs more work on prompt quality to compete with direct code generation.

Both approaches share the same insight: *a prompt guides, a grammar enforces*. For a benchmark with a hard correctness criterion, the guarantee matters.
