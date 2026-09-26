"""
Address normalization.

Evidence-driven design (see reports/eda_summary.md, sampled directly from
real matched pairs in train_ground_truth.tsv):

  - Field order is NOT reliable. One genuine match has business_address
    "AZ, Fl 1st Floor, Surprise, 17437 Guthrie Street" -- state first,
    street number last. A positional parser (number, street, city, state,
    zip in fixed slots) would silently mis-parse rows like this.
    => address_tokens is a BAG of tokens, not a positional parse. Positional
       parsing is out of scope here (Person C can build positional features
       on top of these tokens if useful, but this module does not gate on it).

  - Numbers drift even in true matches ("3928 14th Avenue" vs
    "3930 14th Avenue" vs "3928 1/2 14TH AVE"). Numeric tokens are extracted
    SEPARATELY from address_numbers so a downstream small-delta-tolerant
    comparison is possible, instead of requiring exact numeric equality.

  - ~3.3% of source2/source3 business_address values are empty (0% in
    source1). address_is_empty is exposed explicitly so downstream
    components can route these records differently (e.g. rely on name-only
    signals) rather than treating a missing address as "no similarity".

  - Common abbreviations appear inconsistently (St/Street, Rd/Road, Ave/
    Avenue). A small expansion table normalizes these into address_tokens
    without touching address_clean (the least-destructive form).
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:/\d+)?")  # handles "1/2" style fractions too
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:/\d+)?", flags=re.UNICODE)  # words OR numbers

# Evidence-based, deliberately small -- extend only from observed data.
_ABBREV_EXPANSIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "hwy": "highway", "sq": "square", "pl": "place", "apt": "apartment",
    "fl": "floor", "no": "number", "ste": "suite",
}


def normalize_unicode(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", s)


def collapse_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def clean_address(raw_address: str) -> str:
    """Least-destructive normalized form: unicode-normalized, whitespace
    collapsed, trimmed. Punctuation and casing preserved."""
    if raw_address is None:
        return ""
    return collapse_whitespace(normalize_unicode(raw_address))


def to_ascii(s: str) -> str:
    if not s:
        return ""
    return unidecode(s)


def extract_numbers(raw_address: str) -> list[str]:
    """Pull out numeric substrings (house numbers, PIN/ZIP codes) so they can
    be compared with small-delta tolerance downstream instead of forcing
    exact string equality against the rest of the address."""
    if not raw_address:
        return []
    return _NUMBER_RE.findall(raw_address)


def tokenize_address(address_clean: str) -> list[str]:
    """
    Bag-of-tokens for the address: lower-cased words and number tokens,
    with common abbreviations expanded. Deliberately UNORDERED semantically
    -- callers should compare as sets/multisets, not by position, because
    real address field order is not reliable (see module docstring).
    """
    if not address_clean:
        return []
    lowered = address_clean.lower()
    raw_tokens = _TOKEN_RE.findall(lowered)
    expanded = [_ABBREV_EXPANSIONS.get(t, t) for t in raw_tokens]
    return expanded
