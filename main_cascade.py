"""FastAPI server — Gemma 4 E2B with cascade generation.

Fork of main.py that uses the fast → constrained → retry cascade defined in
gemma_cascade.py. Same HTTP contract as main.py (POST /chat with
{message, tables} → {response}) so it's a drop-in replacement for the
polars.bench runner.

Usage (on the VM):
    uv run uvicorn main_cascade:app --host 0.0.0.0 --port 9000

Env toggles (optional):
    POLARIS_DISABLE_CONSTRAINED=1  force skip the L2 grammar step
"""
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from gemma_cascade import CONSTRAINED_AVAILABLE, run_cascade
from gemma_model import DEFAULT_MODEL_NAME, GemmaModel

MODEL_NAME = os.environ.get("POLARIS_MODEL_NAME", DEFAULT_MODEL_NAME)
DISABLE_CONSTRAINED = os.environ.get("POLARIS_DISABLE_CONSTRAINED", "").lower() in (
    "1", "true", "yes",
)

app = FastAPI()

# Loaded once at startup; reused across requests.
gemma = GemmaModel(MODEL_NAME)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "cascade": True,
        "constrained_available": CONSTRAINED_AVAILABLE,
        "constrained_disabled_via_env": DISABLE_CONSTRAINED,
    }


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    result = run_cascade(
        gemma,
        payload.message,
        payload.tables,
        disable_constrained=DISABLE_CONSTRAINED,
    )
    # Visible in RunPod container stdout — useful for post-mortem analysis.
    print(
        f"[cascade] level={result.level} "
        f"reason={result.reason!r} "
        f"l1_reason={result.l1_reason!r} "
        f"code_len={len(result.code)}"
    )
    return ChatResponse(response=result.code)
