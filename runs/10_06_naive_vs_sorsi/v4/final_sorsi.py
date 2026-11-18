"""Generation 5: PII redactor with improved NAME patterns for accents, hyphens, and exclusion of verb phrases.
"""
import re

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.]+@[\w]+\.\w+")),
    # PHONE requires international prefix, or parentheses around area code, or dashes/dots/spaces separating groups, or preceded by phone/call/text keywords.
    ("PHONE", re.compile(r"\b(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}|\+?\d{1,3}[-.\s]\d{3,4}[-.\s]\d{3,4}[-.\s]\d{3,4}|(?:phone|call|text|cell|mobile)(?:\s+(?:me\s+at|number|is)?)?\s*:?\s*)?\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x)\.?\s*\d+)?\b", re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*){13,19}\b")),
    # ADDRESS: Full US-style postal address matching street number, street name, optional unit/apt, city, state, and ZIP/ZIP+4
    ("ADDRESS", re.compile(
        r"\b\d{1,6}\s+[A-Za-z0-9\.\#\-\s,]+(?:Apt\.?|Suite|Unit|Building|Rm\.?)\s*\d+[A-Za-z0-9\-\s,]*,\s+[A-Za-z\s\-\.]+,?\s+(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b"
        r"|"
        r"\b\d{1,6}\s+[A-Za-z0-9\.\#\-\s]+(?:\r?\n|\s+)[A-Za-z\s\-\.]+,?\s+(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b"
        r"|"
        r"\b\d{1,6}\s+[A-Za-z0-9\.\#\-]+(?:\s+[A-Za-z0-9\.\#\-]+){1,5},?\s+[A-Za-z\s\-\.]+,?\s+(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b",
        re.IGNORECASE
    )),
]

NAME_PATTERNS = [
    # Introduction: "my name is X", "I am X", "This is X writing in", "hey its X" (excluding "talking about")
    re.compile(r"\b(?:my name is|I am|I'm|This is|hey its)\s+(?!talking about)([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+(?:van\s+)?[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+){1,3})\b"),
    # Third party references: "my wife X", "my husband X", "my colleague X", "my son X", "my daughter X", "my manager X", "my partner X"
    re.compile(r"\b(?:my\s+(?:wife|husband|colleague|son|daughter|manager|partner))\s+([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+(?:\s+(?:van\s+)?[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+){1,3})\b"),
    # Sign-offs: "Regards,\n<Name>", "Thanks,\n<Name>", "- <Name>"
    re.compile(r"(?:Regards,|Thanks,|Sincerely,|Best,)\s*\n+\s*([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+(?:\s+(?:van\s+)?[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+){1,2})\b"),
    re.compile(r"\n\s*-\s*([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\-]+)?)\b"),
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
