"""
text_utils.py
-------------
Reusable preprocessing components shared by the training notebook and the
inference notebook. Keeping them in a module (instead of inside the notebook)
lets the pickled pipeline be loaded anywhere: pickle stores a reference to
`text_utils.ReviewPreprocessor`, so this file must sit next to the notebook
(or be importable) when the model is loaded.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
# Negations and contrast words carry sentiment ("not good", "but"), so they are
# removed from the default stop-word list.
KEEP_WORDS = {"no", "not", "nor", "never", "nothing", "none", "cannot", "but",
              "however", "very", "too", "only", "against", "without", "off", "down", "up"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS

SLANG_MAP = {
    "gr8": "great", "u": "you", "ur": "your", "tbh": "to be honest", "rly": "really", "w/": "with",
    "bc": "because", "tho": "though", "pls": "please", "prob": "probably",
    "won't": "will not", "can't": "cannot", "n't": " not", "i'm": "i am", "it's": "it is",
    "you'd": "you would", "don't": "do not", "didn't": "did not", "doesn't": "does not",
}

EMOJI_MAP = {
    "😍": " emojipositive ", "👍": " emojipositive ", "🔥": " emojipositive ", "❤️": " emojipositive ",
    "❤": " emojipositive ", "😊": " emojipositive ",
    "😡": " emojinegative ", "👎": " emojinegative ", "🤬": " emojinegative ", "😤": " emojinegative ",
    "💩": " emojinegative ",
    "😐": " emojineutral ", "🤷": " emojineutral ", "🙂": " emojineutral ",
}

CHANNEL_MAP = {
    "official webstore": "Official Webstore", "webstore": "Official Webstore",
    "amazon": "Amazon", "amazon.com": "Amazon",
    "walmart": "Walmart Marketplace", "walmart marketplace": "Walmart Marketplace",
    "bestbuy": "Best Buy Marketplace", "best buy": "Best Buy Marketplace",
    "best buy marketplace": "Best Buy Marketplace",
}

VERIFIED_MAP = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
                "no": "No", "n": "No", "false": "No", "0": "No"}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def standardize_channel(value):
    """Map inconsistent channel spellings ('AMAZON', ' Amazon', 'amazon.com') to one label."""
    if pd.isna(value):
        return "Unknown"
    return CHANNEL_MAP.get(str(value).strip().lower(), str(value).strip())


def standardize_verified(value):
    """Map Yes/Y/TRUE/1 and No/N/FALSE/0 to 'Yes' / 'No'; anything else -> 'Unknown'."""
    if pd.isna(value):
        return "Unknown"
    return VERIFIED_MAP.get(str(value).strip().lower(), "Unknown")


def clean_text(text):
    """Normalize a raw review into a clean, model-ready string."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)                      # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # URLs
    for emo, token in EMOJI_MAP.items():                      # emojis -> sentiment tokens
        text = text.replace(emo, token)
    text = text.lower()
    for slang, norm in SLANG_MAP.items():                     # slang & contractions
        if slang in ("n't",):
            text = text.replace(slang, norm)
        else:
            text = re.sub(rf"(?<![\w]){re.escape(slang)}(?![\w])", norm, text)
    text = re.sub(r"(\w)\1{2,}", r"\1", text)                 # "sooo" -> "so"
    text = re.sub(r"[^a-z\s]", " ", text)                     # punctuation / digits
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def count_words(text):
    return len(str(text).split()) if isinstance(text, str) else 0


def uppercase_ratio(text):
    """Share of alphabetic characters written in upper case (a proxy for 'shouting')."""
    if not isinstance(text, str):
        return 0.0
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


# ---------------------------------------------------------------------------
# Custom scikit-learn transformers
# ---------------------------------------------------------------------------
class ReviewPreprocessor(BaseEstimator, TransformerMixin):
    """
    Turns RAW review records into model features, so the saved pipeline can
    receive data exactly as it comes from the client's systems.

    Adds:
      clean_text      - normalized title + review text
      word_count      - number of words in the raw review
      exclaim_count   - number of '!' characters
      upper_ratio     - share of upper-case letters
    and standardizes `verified_purchase` / `sales_channel`.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        title = X["review_title"].fillna("") if "review_title" in X else ""
        body = X["review_text"].fillna("")
        full = (title + ". " + body).str.strip()
        X["clean_text"] = full.apply(clean_text)
        X["word_count"] = body.apply(count_words)
        X["exclaim_count"] = body.str.count("!")
        X["upper_ratio"] = body.apply(uppercase_ratio)
        if "verified_purchase" in X:
            X["verified_purchase"] = X["verified_purchase"].apply(standardize_verified)
        if "sales_channel" in X:
            X["sales_channel"] = X["sales_channel"].apply(standardize_channel)
        return X


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps (winsorizes) numeric columns at Q1 - fold*IQR and Q3 + fold*IQR.
    Boundaries are learned on the training data only, then reused at inference.
    """

    def __init__(self, fold=1.5):
        self.fold = fold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.fold * iqr).values
        self.upper_ = (q3 + self.fold * iqr).values
        # if IQR is 0 (very skewed count data) fall back to the 99th percentile
        p99 = X.quantile(0.99).values
        self.upper_ = np.where(iqr.values == 0, p99, self.upper_)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
