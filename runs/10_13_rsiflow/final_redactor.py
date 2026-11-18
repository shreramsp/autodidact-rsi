"""Generation 8: improved PII redactor fixing phone regex and address conflicts to capture unhyphenated 10-digit and international phone numbers.
"""
import re

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

PATTERNS = [
    ("EMAIL", re.compile(
        r"(?:[\w.+]+(?:\s*\[at\]\s*|\s+at\s+)[\w.]+(?:\s*\[dot\]\s*|\s+dot\s+)[\w.]+)|"
        r"(?:[\w.+]+@[\w]+\.[\w.]+)",
        re.IGNORECASE
    )),
    ("PHONE", re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x)\.?\s*\d+)?|"
        r"\b\d{10}\b|"
        r"\+\d{1,3}\s+\d{4,5}\s+\d{4,5}"
    )),
    ("SSN", re.compile(
        r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
    )),
    ("CREDIT_CARD", re.compile(
        r"\b(?:\d[ -]*){13,19}\b"
    )),
    ("ADDRESS", re.compile(
        r"\d+\s+[A-Za-z0-9\s.,#-]+(?:Apt|Suite|Unit|Flat|Department)?\.?\s*\d*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?|"
        r"\d+\s+[A-Za-z0-9\s.,#-]+(?:Apt|Suite|Unit|Flat|Department)?\.?\s*\d*\s*\n\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?|"
        r"\d+\s+[A-Za-z0-9\s.,#-]+(?:Apt|Suite|Unit|Flat|Department)?\.?\s*\d*\s+[A-Za-z\s]+\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?",
        re.IGNORECASE
    )),
    ("NAME", re.compile(
        r"(?:(?:my name is|I am|I'm|this is|husband|wife|colleague|friend|name is|contact|son)\s+([A-Z][a-zà-öø-ÿ]+(?:\s+(?:[dD]e|[dD]en|[vV]an|[vV]on|[dD]u|[vV]ande)?\s*[A-Z][a-zà-öø-ÿ\'-]+)+))|"
        r"(?:(?:Thanks|Regards|Best|Sincerely),\s*\n*([A-Z][a-zà-öø-ÿ]+(?:\s+(?:[dD]e|[dD]en|[vV]an|[vV]on|[dD]u|[vV]ande)?\s*[A-Z][a-zà-öø-ÿ\'-]+)+))|"
        r"(?:hey\s+its\s+([a-zà-öø-ÿ]+(?:\s+[a-zà-öø-ÿ]+)+))|"
        r"(?:^|\n)\s*-\s*([A-Z][a-zà-öø-ÿ]+(?:\s+[A-Z][a-zà-öø-ÿ\'-]+)?)\s*(?:\n|$)",
        re.IGNORECASE
    )),
]


def find_pii(text):
    """Return a list of (start, end, label) spans that contain PII."""
    spans = []
    
    # 1. Standard pattern matching
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if label == "CREDIT_CARD":
                raw_cc = m.group(0)
                if not luhn_check(raw_cc):
                    continue
            
            # Avoid matching part numbers or order numbers near keywords
            if label in ("PHONE", "SSN"):
                start_ctx = max(0, m.start() - 30)
                ctx = text[start_ctx:m.start()].lower()
                if "part number" in ctx or "order number" in ctx or "tracking number" in ctx or "order #" in ctx:
                    continue

            # For NAME pattern, capture groups might be used
            if label == "NAME":
                found_span = None
                for g_idx in range(1, m.lastindex + 1 if m.lastindex else 1):
                    if m.group(g_idx):
                        start = m.start(g_idx)
                        end = m.end(g_idx)
                        found_span = (start, end, label)
                        break
                if found_span:
                    spans.append(found_span)
                else:
                    spans.append((m.start(), m.end(), label))
            else:
                spans.append((m.start(), m.end(), label))

    # Clean overlapping spans (prefer longer or specific types)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = 0
    for start, end, label in spans:
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end
        elif end > last_end:
            # Overlap handling: keep the one that started earlier or is longer
            if filtered and (start < filtered[-1][0] or (end - start) > (filtered[-1][1] - filtered[-1][0])):
                filtered[-1] = (start, end, label)
                last_end = end

    return filtered
