"""Gemma 4 E2B wrapper for Polars code generation.

Extracted from benchmark_gemma.py so both the local benchmark and the FastAPI
server (main_cascade.py) can share the same model implementation without
duplication.

Three generation modes, all operating on the same chat-format prompt built
from gemma_prompt.SYSTEM_PROMPT + FEWSHOT + the current (tables, question):

  - generate(...)              : greedy decoding, fast (~2-3s)
  - generate_constrained(...)  : Outlines CFG on top of the Polars grammar
                                  (slower, ~5-8s, but structurally valid by
                                  construction). Requires `llguidance`.
  - generate_with_feedback(...): appends the failing previous attempt + a
                                  feedback user turn, then re-generates
                                  greedily. Useful as a retry fallback.

Text-only path (AutoTokenizer), no multimodal deps required.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.polars_grammar import build_grammar
from gemma_prompt import FEWSHOT, SYSTEM_PROMPT, format_user_turn

DEFAULT_MODEL_NAME = "google/gemma-4-E2B-it"


def strip_code_fence(text: str) -> str:
    """Strip optional ```python ... ``` fences and surrounding whitespace."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class GemmaModel:
    def __init__(self, name: str = DEFAULT_MODEL_NAME):
        print(f"Loading {name}...")
        # Text-only path: AutoTokenizer avoids pulling the multimodal chain
        # (VideoProcessor -> torchvision). We never send images/audio here.
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self._outlines_model = None  # lazy: only loaded on first constrained call

    def _base_messages(self, message: str, tables: dict) -> list[dict]:
        """Common prefix: system + few-shot examples + the current user turn."""
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_tables, fs_q, fs_a in FEWSHOT:
            messages.append({"role": "user", "content": format_user_turn(fs_tables, fs_q)})
            messages.append({"role": "assistant", "content": fs_a})
        messages.append({"role": "user", "content": format_user_turn(tables, message)})
        return messages

    def _render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_prompt(self, message: str, tables: dict) -> str:
        return self._render(self._base_messages(message, tables))

    def _ensure_outlines(self) -> None:
        if self._outlines_model is None:
            import outlines
            self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)

    @torch.inference_mode()
    def _greedy_from_text(self, text: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return strip_code_fence(response)

    def generate(self, message: str, tables: dict, max_new_tokens: int = 10000) -> str:
        """Fast greedy generation with the prompt v2 + few-shot context."""
        text = self._build_prompt(message, tables)
        return self._greedy_from_text(text, max_new_tokens)

    @torch.inference_mode()
    def generate_constrained(
        self, message: str, tables: dict, max_new_tokens: int = 10000
    ) -> str:
        """Grammar-constrained generation via Outlines CFG.

        Builds a per-request Polars grammar with the request's table/column
        names injected so the decoder can only emit structurally-valid Polars
        method chains over real tables/columns — physically prevents API
        hallucinations like `with_column` or `pl.desc`.

        Raises ImportError (from _ensure_outlines) if `outlines`/`llguidance`
        are not installed. The cascade caller should catch this and fall
        through to the next level.
        """
        from outlines.types import CFG
        self._ensure_outlines()
        prompt = self._build_prompt(message, tables)
        grammar = build_grammar(tables)
        response = self._outlines_model(
            prompt, CFG(grammar), max_new_tokens=max_new_tokens
        )
        return strip_code_fence(response)

    def generate_with_feedback(
        self,
        message: str,
        tables: dict,
        previous_code: str,
        feedback: str,
        max_new_tokens: int = 10000,
    ) -> str:
        """Retry generation after a failed attempt.

        Appends the bad attempt as an assistant turn, then a user turn asking
        for a fix with the feedback text. Greedy decoding, same context as the
        fast path — no grammar involved.
        """
        messages = self._base_messages(message, tables)
        messages.append({"role": "assistant", "content": previous_code or ""})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous code has the following issues: {feedback}. "
                "Rewrite the code using only the modern Polars APIs listed in "
                "the system prompt. Keep it concise and assign the final "
                "DataFrame to `result`. Return only the corrected code, no prose."
            ),
        })
        text = self._render(messages)
        return self._greedy_from_text(text, max_new_tokens)
