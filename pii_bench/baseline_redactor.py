"""Generation 0: a deliberately simple PII redactor.

Every RSI method in this repo starts from this exact file and tries to
improve it. It only knows the most obvious formats and has no idea about
names or addresses.
"""
import re

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.]+@[\w]+\.\w+")),
    ("PHONE", re.compile(r"\d{3}-\d{3}-\d{4}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b\d{16}\b")),
]


def find_pii(text):
    """Return a list of (start, end, label) spans that contain PII."""
    spans = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label))
    return spans
