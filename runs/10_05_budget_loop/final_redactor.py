"""Enhanced PII redactor with name recognition, improved phone, credit card, SSN, and address detection."""
import re
import string

# Luhn checksum for credit cards
def luhn_check(card_num: str) -> bool:
    digits = [int(c) for c in card_num if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    double = False
    for d in reversed(digits):
        if double:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
        double = not double
    return checksum % 10 == 0

# Name patterns based on contexts like "my name is X", "I'm X", sign-offs ("Regards,\nX"), or relative references ("my wife X")
NAME_PATTERNS = [
    # Hello, my name is [Name] / this is [Name] / I'm [Name]
    re.compile(r"\b(?:my name is|this is|I'm|I am)\s+([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){1,2})\b", re.IGNORECASE),
    # My wife / husband / colleague / son / daughter [Name]
    re.compile(r"\b(?:wife|husband|colleague|son|daughter|partner|friend)\s+([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){1,2})\b", re.IGNORECASE),
    # Sign-offs: Regards, / Thanks, / Sincerely, \n [Name]
    re.compile(r"(?:Regards|Thanks|Sincerely|Best|Cheers)[,\s]+\n+([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){0,2})\b"),
    # Dash sign-offs: \n - [Name]
    re.compile(r"\n\s*-\s*([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){0,2})\b"),
]

# US States and territories abbreviation list for address matching
US_STATES = r"(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)"

ADDRESS_PATTERN = re.compile(
    r"\b\d{2,6}\s+"  # Street number
    r"(?:[A-Z0-9a-zà-öø-ÿ\.\-]+\s+){1,5}?"  # Street name and type
    r"(?:(?:Apt|Suite|Unit|Ste|Room|Rm)\.?\s*[\d\-A-Za-z]+,?\s*)?"  # Optional Apt/Suite
    r"(?:[\s,]+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ\.\-]+)+,?\s+"  # City
    r"(?:" + US_STATES + r")\s+"  # State
    r"\d{5}(?:-\d{4})?\b",  # ZIP code
    re.IGNORECASE
)

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.]+@[\w]+\.\w+|[\w.]+\s*\[\s*at\s*\]\s*[\w.]+\s*\[\s*dot\s*\]\s*\w+")),
    ("PHONE", re.compile(
        r"(?:"
        r"\b(?:\+?1[.-]?)?\s*(?:\(\s*\d{3}\s*\)|\d{3})[.-]\d{3}[.-]\d{4}\b"  # Formatted US phone numbers with separators
        r"|\b\+\d{1,3}\s+\d{2,4}\s+\d{3,4}\s+\d{3,4}\b"  # International phone numbers with spaces
        r"|(?:phone|call|text|mob|cell)(?:[:\s]+\w+){0,2}?[:\s]*(\+?\d[\d\s().-]{8,}\d)"  # Numbers near phone context
        r"|(?:best\s+phone)[:\s]*(\+?\d[\d\s().-]{8,}\d)"  # "Best phone:" followed by number
        r")",
        re.IGNORECASE
    )),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s+\d{2}\s+\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*){13,19}\b")),
    ("ADDRESS", ADDRESS_PATTERN),
]


def find_pii(text):
    """Return a list of (start, end, label) spans that contain PII."""
    spans = []
    
    # Standard patterns
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if label == "PHONE":
                # If there are capturing groups and one matched, use that group
                group_matched = False
                if m.lastindex and m.lastindex >= 1:
                    for i in range(1, m.lastindex + 1):
                        if m.start(i) != -1:
                            start = m.start(i)
                            end = m.end(i)
                            val = m.group(i)
                            group_matched = True
                            break
                if not group_matched:
                    # If it matched the general phone pattern, ensure it has a phone context or explicit standard formatting
                    val = m.group(0)
                    start = m.start()
                    end = m.end()
                    # Check preceding context for order/part number indicators
                    prefix = text[max(0, start - 25):start].lower()
                    if any(kw in prefix for kw in ["order", "part", "model", "tracking", "card"]):
                        continue
            else:
                start = m.start()
                end = m.end()
                val = m.group(0)

            if label == "CREDIT_CARD":
                digits_only = "".join(c for c in val if c.isdigit())
                if not luhn_check(digits_only):
                    continue
            spans.append((start, end, label))

    # Name patterns
    for pattern in NAME_PATTERNS:
        for m in pattern.finditer(text):
            # If the pattern has a capturing group for the name itself, use that group
            if m.lastindex and m.lastindex >= 1:
                start = m.start(1)
                end = m.end(1)
                name_str = m.group(1)
            else:
                start = m.start()
                end = m.end()
                name_str = m.group(0)
            
            # Avoid false positives like common words or standalone cities/products if captured incorrectly
            if name_str.lower() in ["apple watch", "galaxy buds", "san francisco", "suite"]:
                continue
            spans.append((start, end, "NAME"))

    # Sort and remove overlaps (keep longer or first)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = 0
    for start, end, label in spans:
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end
        elif end > last_end:
            # Overlap handling: if new extends further, check if it's better
            pass

    return filtered
