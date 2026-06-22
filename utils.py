# Preprocessing text yang di input di app

import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


_RE_URL       = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION   = re.compile(r"@\w+")
_RE_HASHTAG   = re.compile(r"#\w+")
_RE_REUTERS   = re.compile(r"^.*?\(reuters\)\s*[-–]?\s*", re.IGNORECASE)
_RE_BRACKETS  = re.compile(r"\[.*?\]|\(.*?\)")
_RE_HTML      = re.compile(r"<[^>]+>|&\w+;")
_RE_NONALPHA  = re.compile(r"[^a-z\s]")
_RE_SPACES    = re.compile(r"\s+")


def preprocessing(text : str) -> str:
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)
    text = _RE_REUTERS.sub(" ", text)
    text = _RE_BRACKETS.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_NONALPHA.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    text = " ".join(cleaned_tokens)
    
    return text