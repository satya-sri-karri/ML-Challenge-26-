"""
Business-name normalization.

Design principle (see PROJECT_CONTEXT.md / EDA findings in
reports/eda_summary.md): NEVER destroy information. Every function here
returns a NEW value; callers are expected to keep the raw business_name
column untouched alongside whatever this module produces.

Evidence-driven decisions (from reports/eda_summary.md, sampled directly
from train_ground_truth.tsv matches):
  - Real matched names contain character-level typos ("Wilblims"/"Williams",
    "Stee1"/"Steel" -- digit-for-letter substitution). We do NOT attempt to
    "fix" typos here; that is a similarity-feature concern for Person C.
    Normalization only removes NON-SEMANTIC noise (case, punctuation,
    whitespace, legal suffixes), not typos.
  - ~11-15% of source2/source3 names are non-ASCII (mostly Indian-language
    scripts). source1 (the reference) is 0% non-ASCII. name_ascii gives a
    same-alphabet fallback for cross-script comparison.
  - ~4% of source2 names are bare domains ("aristosteel.com"). These break
    every token-based comparison unless the TLD/domain shape is handled
    explicitly.
  - Real matches include truncated names ("Atlantic" matching
    "Atlantic Pinnacle Deli Inc."). We do not attempt to compensate for that
    here -- containment/substring features are a Person C concern -- but we
    do make sure name_tokens preserves individual words so that a downstream
    containment check is possible.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from unidecode import unidecode

from .schema import LEGAL_SUFFIXES

_WS_RE = re.compile(r"\s+")
_PUNCT_TO_SPACE_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DOMAIN_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-]*\.(com|in|net|org|co|biz|info)(\.[a-z]{2})?$"
)
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_unicode(s: str) -> str:
    """NFKC-normalize a string. Safe, non-destructive (no casing/removal)."""
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", s)


def collapse_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def clean_name(raw_name: str) -> str:
    """
    Unicode-normalize + whitespace-collapse a raw business name.
    Casing and script are preserved -- this is the least-destructive
    normalized form, safe to use as the base for every other derived field.
    """
    if raw_name is None:
        return ""
    return collapse_whitespace(normalize_unicode(raw_name))


def to_ascii(s: str) -> str:
    """Transliterate to ASCII (unidecode). Used only as a FALLBACK comparison
    field -- the original-script value is always kept in name_clean/name_lower."""
    if not s:
        return ""
    return unidecode(s)


def is_domain_style(raw_name: str) -> bool:
    """
    True if the raw name looks like a bare domain, e.g. "aristosteel.com",
    "wilfordhancock.com". Observed in ~4% of source2 business_name values
    (see reports/eda_summary.md). Deliberately conservative (anchored regex)
    to avoid flagging names that merely CONTAIN a dot.
    """
    if not raw_name:
        return False
    candidate = raw_name.strip().lower()
    # allow a leading "www." the same way a domain would
    candidate = re.sub(r"^www\.", "", candidate)
    return bool(_DOMAIN_RE.match(candidate))


def clean_domain_name(raw_name: str) -> Optional[str]:
    """
    For a name that is domain-style, strip the TLD and attempt to split
    concatenated/camelCase words back into tokens, e.g.:
        "aristosteel.com"        -> "aristo steel"    (best-effort, may stay
        "ARISTOSTEEL.COM"        -> "aristosteel"       concatenated if no
        "wilfordhancock.com"     -> "wilfordhancock"    case/word boundary
                                                          signal exists)
    Returns None if raw_name is not domain-style -- callers should check
    is_domain_style() first (this function does not re-check).
    Word-splitting is best-effort only: even left concatenated, the string
    still works fine as input to character n-gram similarity features
    downstream, so we do not attempt aggressive dictionary-based segmentation.
    """
    if not raw_name:
        return None
    candidate = re.sub(r"^www\.", "", raw_name.strip(), flags=re.IGNORECASE)
    candidate = re.sub(r"\.(com|in|net|org|co|biz|info)(\.[a-z]{2})?$", "", candidate,
                        flags=re.IGNORECASE)
    # split camelCase boundaries if present (rare but free to do)
    candidate = _CAMEL_SPLIT_RE.sub(" ", candidate)
    candidate = candidate.replace("-", " ").replace("_", " ")
    return collapse_whitespace(candidate).lower()


def strip_legal_suffix(name_lower: str):
    """
    Detect and remove a trailing legal-entity suffix using the small,
    evidence-based table in schema.LEGAL_SUFFIXES (extend only on real
    observed evidence, not speculatively).

    Returns (name_core, suffix_norm_or_None). Only strips from the END of
    the name and only exact-matches a suffix token/phrase after stripping
    trailing punctuation -- deliberately conservative so it never eats part
    of the actual business name.
    """
    if not name_lower:
        return name_lower, None

    working = name_lower.strip()
    # try the longest keys first (e.g. "pvt ltd" before "ltd")
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        pattern = r"[\s,.\-]*" + re.escape(suffix) + r"[\s.\-]*$"
        if re.search(pattern, working):
            core = re.sub(pattern, "", working).strip(" ,.-")
            return collapse_whitespace(core), LEGAL_SUFFIXES[suffix]
    return working, None


def tokenize_name(name_core: str) -> list[str]:
    """Split into a bag of word tokens (order not preserved as meaningful --
    downstream token-set comparison, not positional matching)."""
    if not name_core:
        return []
    stripped = _PUNCT_TO_SPACE_RE.sub(" ", name_core)
    return [t for t in stripped.split() if t]
