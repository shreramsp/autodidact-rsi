"""Fast checks for the shared foundation (no API calls).  Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_bench import generator
from pii_bench.budget import LinearReadout
from pii_bench.editing import diff_features, parse_response
from pii_bench.generator import Message, Span, _luhn_ok
from pii_bench.metrics import score
from pii_bench.sandbox import check_code, RejectedCode, run_code
from pii_bench.sorsi import Attempt, detect

import pytest


def msg(text, *spans):
    return Message("t", text, [Span(s, e, lab, "x") for s, e, lab in spans])


def test_generator_offsets_match_text():
    for m in generator.generate(50, seed=7, prefix="t"):
        for s in m.spans:
            assert 0 <= s.start < s.end <= len(m.text)
            assert m.text[s.start:s.end].strip() == m.text[s.start:s.end]


def test_generator_cards_pass_luhn_and_decoys_do_not():
    for m in generator.generate(200, seed=3, prefix="t"):
        for s in m.spans:
            if s.label == "CREDIT_CARD":
                assert _luhn_ok("".join(c for c in m.text[s.start:s.end] if c.isdigit()))


def test_metrics_perfect_partial_and_false_alarm():
    m = msg("Call 408-555-0199 now", (5, 17, "PHONE"))
    assert score([m], [[(5, 17, "PHONE")]]).f1 == 1.0
    half = score([m], [[(5, 10, "PHONE")]])          # covers < 80% -> a leak, not a hit
    assert half.tp == 0 and half.fn == 1
    fa = score([m], [[(5, 17, "PHONE"), (0, 4, "NAME")]])
    assert fa.tp == 1 and fa.fp == 1


def test_crash_is_scored_as_found_nothing():
    m = msg("x 123-45-6789", (2, 13, "SSN"))
    rep = score([m], [None])
    assert rep.errors == 1 and rep.fn == 1


@pytest.mark.parametrize("bad", ["import os\ndef find_pii(t): return []",
                                 "def find_pii(t): return eval('[]')",
                                 "def find_pii(t): return ().__class__",
                                 "x = 1"])
def test_sandbox_rejects_unsafe_or_incomplete_code(bad):
    with pytest.raises(RejectedCode):
        check_code(bad)


def test_sandbox_runs_good_code_and_times_out_bad_code():
    good = "import re\ndef find_pii(t):\n    return [(m.start(), m.end(), 'SSN') for m in re.finditer(r'\\d{3}-\\d{2}-\\d{4}', t)]\n"
    assert run_code(good, ["my ssn 123-45-6789"]).predictions == [[[7, 18, "SSN"]]]
    slow = "def find_pii(t):\n    while True:\n        pass\n"
    assert run_code(slow, ["x"], timeout=2).status == "timeout"


def test_parse_response_and_diff_features():
    text = 'INTENT: {"target": "PHONE", "change": "c", "expected": "e"}\n```python\ndef find_pii(t):\n    return []\n```'
    code, intent = parse_response(text)
    assert intent["target"] == "PHONE" and "def find_pii" in code
    f = diff_features("import re\nA = re.compile(r'\\d+')\n", "import re\nA = re.compile(r'\\d{3}')\ndef g():\n    pass\n")
    assert f[0] >= 2 and f[2] == 1 and f[3] == 1 and f[4] == 1   # lines added, regex +1/-1, func +1


def test_linear_readout_learns_a_linear_signal():
    rows = [{"a": i, "b": (i * 7) % 5, "y": 2 * i + 1} for i in range(20)]
    m = LinearReadout(ridge=1e-6, features=["a", "b"]).fit(rows, target="y")
    assert abs(m.predict([{"a": 30, "b": 0}])[0] - 61) < 0.5


def _att(r, dp, dr, accepted=False, target="PHONE"):
    deltas = {lab: {"dp": 0, "dr": 0, "df1": 0} for lab in ("NAME", "EMAIL", "PHONE", "CREDIT_CARD", "SSN", "ADDRESS")}
    deltas[target] = {"dp": dp, "dr": dr, "df1": (dp + dr) / 2}
    return Attempt(r, target, "c", "e", accepted, 0.5, 0.5, deltas, [], [])


def test_detector_flags_flip_flop_and_repeats():
    h = [_att(1, +0.10, -0.10)]
    assert "opposing" in detect(_att(2, -0.10, +0.10), h)
    h = [_att(1, 0, 0), _att(2, 0, 0)]
    assert "recurrence" in detect(_att(3, 0, 0), h)
    assert detect(_att(1, +0.2, +0.2, accepted=True), []) == []
