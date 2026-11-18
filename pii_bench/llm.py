"""Gemini client with a response cache and a cost meter.

* Cache  - every response is stored in llm_cache/<hash>.json, keyed by
           (model, prompt, temperature, variant). Re-running a lab replays
           the exact same responses for $0, so results are reproducible and
           anyone can rerun the labs without an API key.
* Meter  - every *paid* call is appended to runs/cost_ledger.jsonl with its
           token counts and dollar cost.

Config (from .env):
  GEMINI_API_KEY   your key
  GEMINI_MODEL     default gemini-3.5-flash-lite
  LLM_OFFLINE=1    never call the API; fail if a response is not cached
  LLM_BUDGET_USD   hard cap on total spend across all runs (default 1.00)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "llm_cache"
LEDGER = ROOT / "runs" / "cost_ledger.jsonl"
load_dotenv(ROOT / ".env")

# USD per 1M tokens (paid tier, text). Source: Gemini API pricing page, Nov 2026.
# Output price includes thinking tokens.
PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.8-flash": (0.75, 3.75),
}
DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
# Hard spending cap across ALL runs (summed from the ledger). Default $1.00.
BUDGET_USD = float(os.getenv("LLM_BUDGET_USD") or "1.00")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool


class OfflineMiss(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    """Hard stop: total paid spend (all runs, from the ledger) would pass the cap."""


class Gemini:
    def __init__(self, model: str = DEFAULT_MODEL, tag: str = ""):
        self.model = model
        self.tag = tag                      # which lab is spending (for the ledger)
        self._client = None
        self.session_cost = 0.0
        self.session_calls = 0
        self.session_cached = 0

    def _api(self):
        if self._client is None:
            from google import genai
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
            self._client = genai.Client(api_key=key)
        return self._client

    def _key(self, prompt: str, temperature: float, variant: int, system: str) -> str:
        blob = json.dumps([self.model, system, prompt, temperature, variant])
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.7,
                 variant: int = 0, max_output_tokens: int = 8192) -> LLMResponse:
        """`variant` lets you ask the same prompt several times and get (and cache)
        different samples - used when proposing K alternative edits."""
        CACHE_DIR.mkdir(exist_ok=True)
        key = self._key(prompt, temperature, variant, system)
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text())
            self.session_cached += 1
            return LLMResponse(d["text"], d["input_tokens"], d["output_tokens"], 0.0, True)
        if os.getenv("LLM_OFFLINE") == "1":
            raise OfflineMiss(f"not cached: {key}")
        spent = total_spent()
        if spent + 0.02 > BUDGET_USD:   # 0.02 = generous upper bound for one call
            raise BudgetExceeded(
                f"Budget stop: ${spent:.4f} spent of the ${BUDGET_USD:.2f} cap "
                f"(runs/cost_ledger.jsonl). Raise LLM_BUDGET_USD in .env only with approval.")

        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens,
                                          system_instruction=system or None,
                                          seed=variant)
        last_err = None
        for attempt in range(6):
            try:
                r = self._api().models.generate_content(model=self.model, contents=prompt, config=cfg)
                break
            except Exception as e:  # rate limits / transient server errors
                last_err = e
                time.sleep(min(60, 2 ** attempt * 2))
        else:
            raise RuntimeError(f"Gemini call failed after retries: {last_err}")

        u = r.usage_metadata
        tin = u.prompt_token_count or 0
        tout = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        pin, pout = PRICES.get(self.model, (0.30, 2.50))
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        text = r.text or ""
        path.write_text(json.dumps({"model": self.model, "text": text, "input_tokens": tin,
                                    "output_tokens": tout, "cost_usd": cost}, indent=1))
        LEDGER.parent.mkdir(exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": self.tag,
                                "model": self.model, "in": tin, "out": tout, "usd": round(cost, 6)}) + "\n")
        self.session_cost += cost
        self.session_calls += 1
        return LLMResponse(text, tin, tout, cost, False)

    def meter(self) -> str:
        return (f"Gemini: {self.session_calls} paid calls (${self.session_cost:.4f}), "
                f"{self.session_cached} replayed from cache ($0)")


def total_spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())
