"""FastAPI server exposed to polars.bench — Gemma 4 E2B with cascade generation.

HTTP contract:
    POST /chat  with {"message": str, "tables": dict}
           ->  {"response": <polars python code>}

Generation uses the fast -> constrained -> retry cascade defined in
gemma_cascade.py. The constrained (L2) level auto-skips if outlines/llguidance
aren't installed, so the server works in any env.

Start locally:
    uvicorn main:app --host 0.0.0.0 --port 9000

Env toggles (optional):
    POLARIS_MODEL_NAME              override the HF model id
    POLARIS_DISABLE_CONSTRAINED=1   force skip of the L2 grammar step
    POLARIS_LOG_FULL_CODE=1         log the full generated code (default: first 500 chars)
"""
import os
import time
import traceback

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from gemma_cascade import CONSTRAINED_AVAILABLE, run_cascade_with_exec_retry
from gemma_model import DEFAULT_MODEL_NAME, GemmaModel

MODEL_NAME = os.environ.get("POLARIS_MODEL_NAME", DEFAULT_MODEL_NAME)
DISABLE_CONSTRAINED = os.environ.get("POLARIS_DISABLE_CONSTRAINED", "").lower() in (
    "1", "true", "yes",
)
LOG_FULL_CODE = os.environ.get("POLARIS_LOG_FULL_CODE", "").lower() in ("1", "true", "yes")
# Exec-retry is DISABLED by default after observing degenerate outputs on prod
# (the model sometimes emits a lone ")" or a tuple-instead-of-str when fed the
# retry feedback). Set POLARIS_MAX_EXEC_RETRIES=N to re-enable with N retries.
MAX_EXEC_RETRIES = int(os.environ.get("POLARIS_MAX_EXEC_RETRIES", "0"))

app = FastAPI()

print("=" * 70)
print(f"[startup] MODEL_NAME          = {MODEL_NAME}")
print(f"[startup] CONSTRAINED_AVAILABLE = {CONSTRAINED_AVAILABLE}")
print(f"[startup] DISABLE_CONSTRAINED   = {DISABLE_CONSTRAINED}")
print(f"[startup] LOG_FULL_CODE         = {LOG_FULL_CODE}")
print("=" * 70, flush=True)

# Loaded once at startup; reused across requests.
gemma = GemmaModel(MODEL_NAME)

print("[startup] model loaded, server ready to serve /chat", flush=True)

_request_counter = 0


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def _summarize_tables(tables: dict) -> str:
    """Compact one-line summary of the request's schema."""
    parts = []
    for name, meta in tables.items():
        cols = meta.get("columns") if isinstance(meta, dict) else meta
        n_cols = len(cols) if isinstance(cols, (dict, list, tuple)) else "?"
        n_rows = meta.get("n_rows") if isinstance(meta, dict) else None
        rows_info = f"/{n_rows}rows" if n_rows is not None else ""
        parts.append(f"{name}({n_cols}cols{rows_info})")
    return ", ".join(parts) if parts else "<no tables>"


def _truncate(text: str, limit: int = 500) -> str:
    text = text or ""
    if LOG_FULL_CODE or len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "cascade": True,
        "constrained_available": CONSTRAINED_AVAILABLE,
        "constrained_disabled_via_env": DISABLE_CONSTRAINED,
        "requests_served": _request_counter,
    }


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    global _request_counter
    _request_counter += 1
    req_id = _request_counter

    t0 = time.time()
    question_preview = payload.message[:200].replace("\n", " ")
    schema_summary = _summarize_tables(payload.tables)

    print(
        f"\n[req #{req_id}] ---------------- INCOMING ----------------\n"
        f"[req #{req_id}] schema : {schema_summary}\n"
        f"[req #{req_id}] question: {question_preview}"
        + ("" if len(payload.message) <= 200 else f" ... [+{len(payload.message) - 200} chars]"),
        flush=True,
    )

    try:
        result = run_cascade_with_exec_retry(
            gemma,
            payload.message,
            payload.tables,
            disable_constrained=DISABLE_CONSTRAINED,
            max_retries=MAX_EXEC_RETRIES,
        )
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(
            f"[req #{req_id}] FATAL : {type(e).__name__}: {e} ({elapsed:.1f}s)\n{tb}",
            flush=True,
        )
        # Return an empty-ish code rather than 500 so polars.bench can continue.
        return ChatResponse(response="result = None")

    elapsed = time.time() - t0
    print(
        f"[req #{req_id}] cascade: level={result.level} "
        f"reason={result.reason!r} "
        f"l1_reason={result.l1_reason!r} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    print(f"[req #{req_id}] code  :\n{_truncate(result.code)}", flush=True)
    if result.level != "fast" and result.l1_code and result.l1_code != result.code:
        print(
            f"[req #{req_id}] (l1 was rejected, original attempt was:)\n"
            f"{_truncate(result.l1_code)}",
            flush=True,
        )

    return ChatResponse(response=result.code)
