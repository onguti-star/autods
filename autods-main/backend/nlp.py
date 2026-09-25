"""
Offline NLP utilities.

No external AI API and no downloaded corpora (e.g. nltk's punkt/stopwords
would need an internet fetch on first use, which this app can't rely on).
Everything here is self-contained: a bundled stopword list plus regex-based
tokenization, in the same spirit as assistant.py and clean_chat.py.
"""
import re
from collections import Counter

import pandas as pd

# A standard, compact English stopword list (no external download required).
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_WORD_RE = re.compile(r"[a-zA-Z']+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMBER_RE = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")

# scikit-learn's TfidfVectorizer tokenizes on \w+ (no apostrophes), so "don't" becomes
# "don" + "t" rather than "don't". Passing STOPWORDS directly triggers a UserWarning
# about that mismatch; this variant is pre-split to match sklearn's own tokenization.
SKLEARN_STOPWORDS = sorted({w.split("'")[0] for w in STOPWORDS} | {w.replace("'", "") for w in STOPWORDS})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stripped of punctuation/numbers."""
    if not isinstance(text, str):
        return []
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def is_text_column(s: pd.Series, min_avg_words: float = 4.0, min_unique_ratio: float = 0.1) -> bool:
    """
    Heuristic for 'this is free-form text' (reviews, comments, descriptions)
    as opposed to a short categorical/label string column (e.g. 'city', 'status').
    Word count is the primary signal — a sentence is a sentence even if it repeats
    (templated responses, scraped duplicates). The uniqueness floor only exists to
    rule out a column that's essentially one constant long string repeated everywhere.
    """
    if pd.api.types.is_numeric_dtype(s):
        return False
    non_null = s.dropna().astype(str)
    non_null = non_null[non_null.str.strip() != ""]
    if len(non_null) < 5:
        return False
    word_counts = non_null.str.split().str.len()
    avg_words = float(word_counts.mean())
    unique_ratio = non_null.nunique() / len(non_null)
    return avg_words >= min_avg_words and unique_ratio >= min_unique_ratio


def text_column_stats(s: pd.Series, top_n: int = 20) -> dict:
    """Word/char length stats, vocabulary size, and top non-stopword words for one text column."""
    non_null = s.dropna().astype(str)
    non_null = non_null[non_null.str.strip() != ""]
    if non_null.empty:
        return {
            "documents": 0, "avg_words": 0, "median_words": 0, "min_words": 0, "max_words": 0,
            "avg_chars": 0, "vocab_size": 0, "top_words": [],
        }

    word_counts = non_null.str.split().str.len()
    char_counts = non_null.str.len()

    all_tokens = []
    for text in non_null:
        all_tokens.extend(tokenize(text))
    meaningful = [t for t in all_tokens if t not in STOPWORDS]
    vocab = set(all_tokens)
    top = Counter(meaningful).most_common(top_n)

    return {
        "documents": int(len(non_null)),
        "avg_words": round(float(word_counts.mean()), 1),
        "median_words": round(float(word_counts.median()), 1),
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
        "avg_chars": round(float(char_counts.mean()), 1),
        "vocab_size": len(vocab),
        "top_words": [{"word": w, "count": c} for w, c in top],
    }


def word_frequency(s: pd.Series, top_n: int = 25, exclude_stopwords: bool = True) -> list[dict]:
    """Top-N word frequency list for one text column, used by the word-frequency chart."""
    counter = Counter()
    for text in s.dropna().astype(str):
        for tok in tokenize(text):
            if exclude_stopwords and tok in STOPWORDS:
                continue
            counter[tok] += 1
    return [{"word": w, "count": c} for w, c in counter.most_common(top_n)]


# ---- text cleaning helpers, used by clean_chat.py ----

def remove_stopwords(text) -> str:
    if pd.isna(text):
        return text
    words = str(text).split()
    return " ".join(w for w in words if w.lower() not in STOPWORDS)


def remove_urls(text) -> str:
    if pd.isna(text):
        return text
    return _WHITESPACE_RE.sub(" ", _URL_RE.sub(" ", str(text))).strip()


def remove_emails(text) -> str:
    if pd.isna(text):
        return text
    return _WHITESPACE_RE.sub(" ", _EMAIL_RE.sub(" ", str(text))).strip()


def remove_punctuation(text) -> str:
    if pd.isna(text):
        return text
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub(" ", str(text))).strip()


def remove_numbers(text) -> str:
    if pd.isna(text):
        return text
    return _WHITESPACE_RE.sub(" ", _NUMBER_RE.sub(" ", str(text))).strip()