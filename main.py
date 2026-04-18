"""FastAPI server — Gemma 4 E2B GGUF with grammar-constrained generation.

Uses llama_cpp (GGUF) + pygbnf to constrain decoding to structurally valid
Polars code for the request's schema. No hallucinated methods or table names.

Usage:
    uv run uvicorn main:app --host 0.0.0.0 --port 9000
"""
import json

from fastapi import FastAPI
from pydantic import BaseModel

from gemma_prompt import SYSTEM_PROMPT, FEWSHOT, format_user_turn
from test_gram.test_gram import DEFAULT_FILE, DEFAULT_REPO, build_gbnf

app = FastAPI()

# Model loaded once at startup; reused across requests.
from llama_cpp import Llama, LlamaGrammar

_llm = Llama.from_pretrained(
    repo_id=DEFAULT_REPO,
    filename=DEFAULT_FILE,
    n_ctx=131072,
    n_gpu_layers=-1,
    verbose=False,
)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": DEFAULT_FILE, "constrained": True}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for fs_tables, fs_q, fs_a in FEWSHOT:
        messages.append({"role": "user", "content": format_user_turn(fs_tables, fs_q)})
        messages.append({"role": "assistant", "content": fs_a})
    messages.append({"role": "user", "content": format_user_turn(payload.tables, payload.message)})

    gbnf = build_gbnf(payload.tables)
    grammar = LlamaGrammar.from_string(gbnf)

    out = _llm.create_chat_completion(
        messages=messages,
        grammar=grammar,
        max_tokens=5120,
        temperature=0.0,
    )
    return ChatResponse(response=out["choices"][0]["message"]["content"].strip())
