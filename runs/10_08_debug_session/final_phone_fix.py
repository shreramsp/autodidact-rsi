import re

# Refined PHONE regex to handle international formats, extensions, and word/context boundaries
PHONE = re.compile(
    r"(?<!\d)(?:"
    r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    r"|\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r")"
    r"(?:(?:\s*(?:ext\.?|x)\s*\d+)?)\b(?!-)",
    re.IGNORECASE,
)

# Negative lookahead/lookbehind patterns or post-filtering to exclude order numbers
ORDER_NUMBER_PATTERN = re.compile(r"(?:order|#)\s*(?:number)?[:#]?\s*(\d{3}-\d{3}-\d{4})", re.IGNORECASE)


def find_pii(text):
    """Finds PII in text, focusing on PHONE numbers while filtering out order numbers."""
    spans = []
    
    # Collect potential order numbers to exclude
    excluded_spans = set()
    for m in ORDER_NUMBER_PATTERN.finditer(text):
        # The group 1 or the whole match for order numbers
        sub_m = m.group(1) if m.groups() else m.group(0)
        start = text.find(sub_m, m.start())
        if start != -1:
            excluded_spans.add((start, start + len(sub_m)))

    for m in PHONE.finditer(text):
        start, end = m.start(), m.end()
        
        # Check overlap with excluded order numbers
        if any(es <= start and end <= ee for es, ee in excluded_spans):
            continue
            
        # Additional sanity check: standalone 10-digit numbers preceded by "order" or "#"
        prefix = text[max(0, start - 15):start].lower()
        if "order" in prefix or "#" in prefix:
            # Likely an order number, skip unless it has explicit phone indicators
            if not any(kw in prefix for kw in ["call", "phone", "cell", "reach", "at", "line"]):
                continue

        spans.append((start, end, "PHONE"))
        
    return spans
