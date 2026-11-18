"""Generation 7: adds robust CREDIT_CARD regex matching with Luhn checksum validation."""
import re

def _passes_luhn(card_str: str) -> bool:
    digits = [int(c) for c in card_str if c.isdigit()]
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

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.]+@[\w]+\.\w+")),
    ("PHONE", re.compile(
        r"(?i)(?:phone|cell|mobile|call(?:\s+me)?(?:\s+at)?|text(?:\s+me)?(?:\s+on)?)\s*[:#]?\s*"
        r"("
        r"(?:\+\d{1,3}\s*)?"
        r"(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"
        r"(?:\s*(?:ext|x)\.?\s*\d+)?"
        r")"
        r"|"
        r"\b(?:\+\d{1,3}\s*)?\(\d{3}\)\s*\d{3}-\d{4}\b"
        r"|"
        r"\b(?:\+\d{1,3}[-.\s]?)?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    )),
    ("SSN", re.compile(
        r"(?i)(?:"
        r"(?:ssn|social security(?:\s+number)?)\s*[:#]?\s*(\b\d{3}[-\s]\d{2}[-\s]\d{4}\b|\b\d{9}\b)"
        r"|"
        r"\b(?:\d{3}-\d{2}-\d{4}|\d{3}\s\d{2}\s\d{4})\b"
        r")"
    )),
    ("CREDIT_CARD", re.compile(
        r"\b(?:"
        r"\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}"
        r"|"
        r"\d{4}[-\s]\d{6}[-\s]\d{5}"
        r"|"
        r"\d{15,16}"
        r")\b"
    )),
    ("ADDRESS", re.compile(
        r"\b\d{1,6}\s+[0-9A-Za-z\u00C0-\u017F\s.,#-]+?\s+"
        r"(?:Apt\.?|Suite|Ste\.?|Unit|Room|Fl\.?|Floor)?\s*#?\d*[0-9A-Za-z\u00C0-\u017F.,-]*[\s,]+"
        r"[A-Za-z\u00C0-\u017F\s.,-]+,\s+"
        r"(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|GU|PR|VI|FM)\s+"
        r"\d{5}(?:-\d{4})?\b"
    )),
    ("NAME", re.compile(
        r"(?i)\b(?:"
        r"(?:(?:my name is|i'm|i am|hey its|hey it's|this is|colleague|husband|wife|son)\s+([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+(?:de\s+la\s+|[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)){1,3}))"
        r"|"
        r"(?:(?:Regards,|Thanks,|Sincerely,|Best,|Cheers,)\s*\n*\s*([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+(?:de\s+la\s+|[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)){1,2}))"
        r"|"
        r"(?:-\s*([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:-[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)?(?:\s+(?:de\s+la\s+|[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)){0,2}))"
        r")"
    )),
]


def find_pii(text):
    """Return a list of (start, end, label) spans that contain PII."""
    spans = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if label == "CREDIT_CARD":
                candidate = m.group(0)
                if not _passes_luhn(candidate):
                    continue
                spans.append((m.start(), m.end(), label))
            elif label == "NAME":
                for group_idx in range(1, m.re.groups + 1):
                    start = m.start(group_idx)
                    if start != -1:
                        end = m.end(group_idx)
                        val = text[start:end].strip()
                        if val.lower() in ("talking about",):
                            break
                        spans.append((start, end, label))
                        break
            elif label == "PHONE":
                if m.re.groups >= 1 and m.start(1) != -1:
                    spans.append((m.start(1), m.end(1), label))
                else:
                    spans.append((m.start(), m.end(), label))
            elif label == "SSN":
                if m.re.groups >= 1 and m.start(1) != -1:
                    spans.append((m.start(1), m.end(1), label))
                else:
                    spans.append((m.start(), m.end(), label))
            else:
                spans.append((m.start(), m.end(), label))
    return spans
