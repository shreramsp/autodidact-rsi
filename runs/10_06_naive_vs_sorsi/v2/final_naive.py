"""Generation 4: PII redactor with refined PHONE pattern to avoid capturing 10-digit order numbers.
"""
import re

PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b|\b[\w.-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\[\s*dot\s*\]\s*\w+\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?:\s*(?:ext|x)\.?\s*\d+)?\b|\b\+?\d{1,3}\s+\d{2,4}\s+\d{3,4}\s+\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*){13,19}\b")),
    ("ADDRESS", re.compile(r"\d{1,5}\s+[A-Z0-9a-zà-öø-ÿ\.\,\#\-\s]+(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b\.?(?:,\s*(?:Apt|Suite|Unit|Building|Ste)\s*\d+[A-Z]?)?(?:,\s*[A-Za-z\s]+),\s*(?:A[KLRZ]|C[AOT]|D[CE]|FL|GA|H[IJK]|I[ADLN|L]|K[SY]|LA|M[ADEINOST|D]|N[CDEHJMVY]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])\s+\d{5}(?:-\d{4})?\b")),
]

NAME_PATTERNS = [
    # Introduction: "my name is X", "I'm X", "This is X writing in", "hey its X"
    re.compile(r"\b(?:my name is|I am|I'm|This is|hey its)\s+(?!talking about\b)([A-Za-zà-öø-ÿ]+(?:\s+[A-Za-zà-öø-ÿ]+){1,3})\b"),
    # Third party references: "my wife X", "my husband X", "my colleague X", "my son X", "my daughter X", "my manager X", "my partner X"
    re.compile(r"\b(?:my\s+(?:wife|husband|colleague|son|daughter|manager|partner))\s+([A-Z][a-zà-öø-ÿ]+(?:\s+(?:van\s+)?[A-Z][a-zà-öø-ÿ]+){1,3})\b"),
    # Sign-offs: "Regards,\n<Name>", "Thanks,\n<Name>", "- <Name>"
    re.compile(r"(?:Regards,|Thanks,|Sincerely,|Best,)\s*\n+\s*([A-Z][a-zà-öø-ÿ]+(?:\s+(?:van\s+)?[A-Z][a-zà-öø-ÿ]+){1,2})\b"),
    re.compile(r"\n\s*-\s*([A-Z][a-zà-öø-ÿ]+(?:\s+[A-Z][a-zà-öø-ÿ]+)?)\b"),
]

def luhn_check(card_num: str) -> bool:
    digits = [int(c) for c in card_num if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def find_pii(text):
    """Return a list of (start, end, label) spans that contain PII."""
    spans = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if label == "CREDIT_CARD":
                candidate = m.group(0)
                if not luhn_check(candidate):
                    continue
            spans.append((m.start(), m.end(), label))

    for pattern in NAME_PATTERNS:
        for m in pattern.finditer(text):
            # If the pattern has capturing groups, use group(1) for the actual name span
            if m.groups():
                start = m.start(1)
                end = m.end(1)
                spans.append((start, end, "NAME"))

    # Sort spans and remove overlaps (preferring earlier/longer ones)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = 0
    for start, end, label in spans:
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end
    return filtered
