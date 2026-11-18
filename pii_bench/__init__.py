"""pii_bench: the shared problem every RSI method in this repo improves on.

The problem: given a customer-support message, find every span of
Personally Identifiable Information (PII) so it can be redacted.

    find_pii("Hi, I'm Sarah Chen, call me at (408) 555-0199")
    -> [(8, 18, "NAME"), (31, 45, "PHONE")]

Modules
-------
generator   synthetic support messages with gold PII spans (and decoys)
metrics     span matching -> precision / recall / F1 / leak rate
sandbox     run untrusted, LLM-written redactor code safely in a subprocess
llm         Gemini client with a response cache and a cost meter
editing     prompts for proposing edits, code extraction, diff features
"""

LABELS = ("NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN", "ADDRESS")
