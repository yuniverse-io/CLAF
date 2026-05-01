import html
import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup


_URL_RE = re.compile(r"http\S+|www\.\S+")
_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([^\w\s])")


def clean_review_text(review_text: Optional[str]) -> Optional[str]:
    """Paper Sec 4.1 minimal preprocessing: strip HTML tags/entities and URLs; normalize whitespace and control chars."""
    if pd.isna(review_text):
        return None
    review_text = str(review_text)
    review_text = BeautifulSoup(review_text, "html.parser").get_text()       # HTML tags
    review_text = html.unescape(review_text)                                 # HTML entities (&amp; → &)
    review_text = _URL_RE.sub("[URL]", review_text)                          # URLs
    review_text = _CONTROL_RE.sub(" ", review_text)                          # control chars (newline, tab, ...)
    review_text = _WHITESPACE_RE.sub(" ", review_text)                       # collapse whitespace
    review_text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", review_text)             # drop space before punctuation
    review_text = review_text.strip()
    return review_text if review_text else None


def is_english(review_text: Optional[str], threshold: float = 0.9) -> bool:
    """True if ≥ `threshold` of chars are ASCII; False for empty / null input."""
    if pd.isna(review_text) or not review_text:
        return False
    return sum(1 for c in review_text if ord(c) < 128) / len(review_text) >= threshold
