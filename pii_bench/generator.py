"""Synthetic customer-support messages with gold PII spans.

Every message is built piece by piece, so we always know the exact
character offsets of each PII value. That gives us a perfect answer key
for free, with no real personal data involved (all values come from Faker).

Two kinds of tricky content make the problem worth improving on:

* Hard formats   - e.g. "sarah [at] gmail [dot] com", "+44 20 7946 0958",
                   lowercase names, addresses split across lines.
* Decoys         - NOT PII but look like it, e.g. an order number that looks
                   like a phone, a tracking number that looks like a card
                   (but fails the Luhn checksum), a part number shaped like
                   an SSN, product names that look like person names.

Decoys create a real precision/recall trade-off: a looser phone regex
catches more phones but starts redacting order numbers.

Labeling guide (what counts as PII here)
----------------------------------------
NAME         a person's name (full, first-only in a signature, or a third party)
EMAIL        any personal email address, including obfuscated forms
PHONE        any phone number, US or international
CREDIT_CARD  a payment card number (Luhn-valid)
SSN          a US Social Security Number
ADDRESS      a full street address (street through ZIP), one span
NOT PII      order / tracking / part numbers, prices, dates, product names,
             company names, city names on their own
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from faker import Faker

NAME_LOCALES = ["en_US", "en_GB", "en_IN", "es_MX", "fr_FR", "de_DE"]


@dataclass
class Span:
    start: int
    end: int
    label: str
    fmt: str  # which surface format produced it (for analysis only)


@dataclass
class Message:
    id: str
    text: str
    spans: list[Span] = field(default_factory=list)
    decoys: list[str] = field(default_factory=list)  # decoy kinds present

    def to_json(self) -> dict:
        return {"id": self.id, "text": self.text,
                "spans": [asdict(s) for s in self.spans], "decoys": self.decoys}

    @staticmethod
    def from_json(d: dict) -> "Message":
        return Message(d["id"], d["text"], [Span(**s) for s in d["spans"]], d.get("decoys", []))


class _Builder:
    """Concatenates text pieces while recording PII offsets."""

    def __init__(self):
        self.parts: list[str] = []
        self.spans: list[Span] = []
        self.pos = 0

    def add(self, s: str):
        self.parts.append(s)
        self.pos += len(s)

    def pii(self, s: str, label: str, fmt: str):
        self.spans.append(Span(self.pos, self.pos + len(s), label, fmt))
        self.add(s)

    def text(self) -> str:
        return "".join(self.parts)


# ---------------------------------------------------------------- formats ---

def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _name(fk: dict[str, Faker], rng: random.Random) -> tuple[str, str]:
    loc = rng.choice(NAME_LOCALES)
    f = fk[loc]
    first, last = f.first_name(), f.last_name()
    return first, last


def _email(rng, fk, first, last) -> tuple[str, str]:
    user = rng.choice([f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}",
                       f"{first}_{last}{rng.randint(1, 99)}", f"{last}.{first}"])
    user = "".join(c for c in user.lower() if c.isalnum() or c in "._")
    domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com",
                         "proton.me", "hotmail.co.uk", "sjsu.edu", "comcast.net"])
    fmt = rng.choices(["plain", "plus", "upper", "obf_brackets", "obf_words"],
                      weights=[55, 12, 8, 15, 10])[0]
    if fmt == "plus":
        return f"{user}+{rng.choice(['orders', 'shop', 'news'])}@{domain}", fmt
    if fmt == "upper":
        return f"{user}@{domain}".upper(), fmt
    if fmt == "obf_brackets":
        d, tld = domain.split(".", 1)
        return f"{user} [at] {d} [dot] {tld}", fmt
    if fmt == "obf_words":
        d, tld = domain.split(".", 1)
        return f"{user.replace('.', ' dot ')} at {d} dot {tld}", fmt
    return f"{user}@{domain}", fmt


def _phone(rng) -> tuple[str, str]:
    a, b, c = str(rng.randint(201, 989)), str(rng.randint(200, 999)), _digits(rng, 4)
    fmt = rng.choices(["paren", "dash", "dot", "space", "plus1", "bare", "intl_uk", "intl_in", "ext"],
                      weights=[22, 22, 10, 8, 12, 8, 7, 6, 5])[0]
    return {
        "paren": f"({a}) {b}-{c}",
        "dash": f"{a}-{b}-{c}",
        "dot": f"{a}.{b}.{c}",
        "space": f"{a} {b} {c}",
        "plus1": f"+1 {a} {b} {c}",
        "bare": f"{a}{b}{c}",
        "intl_uk": f"+44 20 {_digits(rng, 4)} {_digits(rng, 4)}",
        "intl_in": f"+91 {rng.randint(70, 99)}{_digits(rng, 3)} {_digits(rng, 5)}",
        "ext": f"{a}-{b}-{c} ext. {rng.randint(10, 999)}",
    }[fmt], fmt


def _card(rng, fk) -> tuple[str, str]:
    kind = rng.choices(["visa", "mastercard", "amex"], weights=[45, 35, 20])[0]
    num = fk["en_US"].credit_card_number(card_type=kind)
    assert _luhn_ok(num)
    if kind == "amex":
        sep = rng.choice([" ", "-", ""])
        return sep.join([num[:4], num[4:10], num[10:]]), f"amex_{'none' if not sep else 'sep'}"
    sep = rng.choices([" ", "-", ""], weights=[45, 35, 20])[0]
    return sep.join(num[i:i + 4] for i in range(0, 16, 4)), f"16_{'none' if not sep else 'sep'}"


def _ssn(rng) -> tuple[str, str]:
    a, b, c = f"{rng.randint(100, 665):03d}", f"{rng.randint(1, 99):02d}", f"{rng.randint(1, 9999):04d}"
    fmt = rng.choices(["dash", "space", "bare"], weights=[60, 20, 20])[0]
    return {"dash": f"{a}-{b}-{c}", "space": f"{a} {b} {c}", "bare": f"{a}{b}{c}"}[fmt], fmt


def _address(rng, fk) -> tuple[str, str]:
    f = fk["en_US"]
    street, city, state, zipc = f.street_address(), f.city(), f.state_abbr(), f.zipcode()
    fmt = rng.choices(["commas", "no_commas", "multiline", "zip4"], weights=[45, 20, 20, 15])[0]
    if fmt == "no_commas":
        return f"{street} {city} {state} {zipc}", fmt
    if fmt == "multiline":
        return f"{street}\n{city}, {state} {zipc}", fmt
    if fmt == "zip4":
        return f"{street}, {city}, {state} {zipc}-{_digits(rng, 4)}", fmt
    return f"{street}, {city}, {state} {zipc}", fmt


# ------------------------------------------------------------- sentences ---

ISSUES = [
    "My package was supposed to arrive on Monday but it still hasn't shown up.",
    "I was charged twice for the same order and need a refund.",
    "The blender I bought stopped working after two days.",
    "I can't log into my account even after resetting the password.",
    "The jacket I received is the wrong size, I ordered a medium.",
    "My subscription renewed even though I cancelled it last month.",
    "The tracking page has said 'label created' for a week now.",
    "Half of the items in my grocery delivery were missing.",
    "I'd like to change the delivery address on my open order.",
    "The app keeps crashing when I try to check out.",
    "I received someone else's order by mistake.",
    "My refund was approved 10 days ago but I haven't seen it yet.",
    "The promo code SAVE20 didn't apply at checkout.",
    "I need an invoice for my company's expense report.",
    "The headphones arrived with a cracked case.",
    "Can you upgrade my plan to the family tier?",
]

DECOY_TEMPLATES = {
    "order_phone_like": lambda r: f"My order number is {r.randint(201, 989)}-{r.randint(200, 999)}-{_digits(r, 4)}.",
    "order_10_digit": lambda r: f"Order #{r.randint(2, 9)}{_digits(r, 9)} is the one I'm talking about.",
    "tracking_card_like": lambda r: f"The tracking number is {_bad_luhn_grouped(r)}.",
    "part_ssn_like": lambda r: f"The replacement part number is {r.randint(100, 899)}-{_digits(r, 2)}-{_digits(r, 4)}.",
    "product_names": lambda r: r.choice([
        "It happened with my Apple Watch and my Galaxy Buds.",
        "The Kindle Paperwhite won't charge anymore.",
        "I bought the Instant Pot Duo during the Black Friday sale.",
        "Both the Google Pixel and the Fire Stick have the same issue.",
        "My Dyson Airwrap arrived without the brush attachment.",
    ]),
    "price_date": lambda r: f"I paid ${r.randint(10, 1999)}.{_digits(r, 2)} on {r.randint(1, 12):02d}/{r.randint(1, 28):02d}/2025.",
    "company_city": lambda r: r.choice([
        "I usually shop at your San Francisco store.",
        "Your Austin warehouse shipped it.",
        "The courier from Fresno Logistics never called.",
    ]),
}


def _bad_luhn_grouped(r: random.Random) -> str:
    while True:
        d = str(r.randint(1, 9)) + _digits(r, 15)
        if not _luhn_ok(d):
            return " ".join(d[i:i + 4] for i in range(0, 16, 4))


def _make_message(idx: int, rng: random.Random, fk: dict[str, Faker], prefix: str) -> Message:
    b = _Builder()
    first, last = _name(fk, rng)
    full = f"{first} {last}"
    decoys: list[str] = []

    # 1) opening (maybe with a name)
    opening = rng.choices(["im", "my_name_is", "this_is", "lower", "none"], weights=[20, 12, 8, 5, 55])[0]
    if opening == "im":
        b.add(rng.choice(["Hi, I'm ", "Hello, I'm ", "Hey there, I'm "])); b.pii(full, "NAME", "intro_im"); b.add(". ")
    elif opening == "my_name_is":
        b.add("Hello, my name is "); b.pii(full, "NAME", "my_name_is"); b.add(" and ")
    elif opening == "this_is":
        b.add("This is "); b.pii(full, "NAME", "this_is"); b.add(" writing in. ")
    elif opening == "lower":
        b.add("hey its "); b.pii(full.lower(), "NAME", "lowercase"); b.add(" again. ")
    else:
        b.add(rng.choice(["Hi, ", "Hello, ", "", "Good morning. "]))

    # 2) the actual complaint
    b.add(rng.choice(ISSUES) + " ")

    # 3) PII sentences (0-3), mixed with decoys (0-2)
    chunks = []
    for kind in rng.sample(["EMAIL", "PHONE", "CREDIT_CARD", "SSN", "ADDRESS", "THIRD_PARTY"], k=rng.choice([0, 1, 2, 2, 3])):
        chunks.append(("pii", kind))
    for kind in rng.sample(list(DECOY_TEMPLATES), k=rng.choice([0, 1, 1, 2])):
        chunks.append(("decoy", kind))
    rng.shuffle(chunks)

    for typ, kind in chunks:
        if typ == "decoy":
            b.add(DECOY_TEMPLATES[kind](rng) + " ")
            decoys.append(kind)
            continue
        if kind == "EMAIL":
            v, fmt = _email(rng, fk, first, last)
            b.add(rng.choice(["You can email me at ", "My email is ", "Reach me at "])); b.pii(v, "EMAIL", fmt); b.add(". ")
        elif kind == "PHONE":
            v, fmt = _phone(rng)
            b.add(rng.choice(["Call me at ", "My number is ", "Text me on ", "Best phone: "])); b.pii(v, "PHONE", fmt); b.add(". ")
        elif kind == "CREDIT_CARD":
            v, fmt = _card(rng, fk)
            b.add(rng.choice(["I paid with the card ", "The card used was ", "Please refund to card "])); b.pii(v, "CREDIT_CARD", fmt); b.add(". ")
        elif kind == "SSN":
            v, fmt = _ssn(rng)
            b.add(rng.choice(["For the credit check my SSN is ", "My social security number is ", "SSN: "])); b.pii(v, "SSN", fmt); b.add(". ")
        elif kind == "ADDRESS":
            v, fmt = _address(rng, fk)
            b.add(rng.choice(["Please ship it to ", "My address is ", "Send the replacement to:\n"])); b.pii(v, "ADDRESS", fmt); b.add(". ")
        elif kind == "THIRD_PARTY":
            f2, l2 = _name(fk, rng)
            b.add(rng.choice(["My wife ", "My husband ", "My colleague ", "My son "])); b.pii(f"{f2} {l2}", "NAME", "third_party")
            b.add(rng.choice([" placed the order. ", " is the account holder. ", " will pick it up. "]))

    # 4) sign-off (maybe with a name)
    sign = rng.choices(["full", "first", "regards", "none"], weights=[15, 10, 10, 65])[0]
    if sign == "full":
        b.add("\n\nThanks,\n"); b.pii(full, "NAME", "signature_full")
    elif sign == "first":
        b.add("\n- "); b.pii(first, "NAME", "signature_first")
    elif sign == "regards":
        b.add("\nRegards, "); b.pii(full, "NAME", "signature_regards")
    else:
        b.add(rng.choice(["Thanks.", "Please help.", "Thank you!", ""]))

    return Message(f"{prefix}-{idx:04d}", b.text().rstrip(), b.spans, decoys)


def generate(n: int, seed: int, prefix: str) -> list[Message]:
    rng = random.Random(seed)
    fk = {}
    for i, loc in enumerate(NAME_LOCALES):
        fk[loc] = Faker(loc)
        fk[loc].seed_instance(seed * 100 + i)
    return [_make_message(i, rng, fk, prefix) for i in range(n)]


# ------------------------------------------------------------- splits ------
#
# feedback  : shown to the LLM (its failures are quoted in prompts)
# steering  : small, cheap split used to score edits during the search
# heldout   : large, never shown to the LLM; the honest final score
SPLITS = {"feedback": (150, 11), "steering": (60, 22), "heldout": (400, 33)}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_all(out_dir: Path = DATA_DIR) -> dict[str, list[Message]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (n, seed) in SPLITS.items():
        msgs = generate(n, seed, name[:2])
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for m in msgs:
                f.write(json.dumps(m.to_json()) + "\n")
        result[name] = msgs
    return result


def load(split: str, data_dir: Path = DATA_DIR) -> list[Message]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        build_all(data_dir)
    with open(path) as f:
        return [Message.from_json(json.loads(line)) for line in f]
