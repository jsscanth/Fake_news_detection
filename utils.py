import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------
# NLTK Resource Safeguard
# -----------------------------------------------------------------------------
# This checks if the NLTK packages exist; if not, it downloads them automatically.
# This prevents crashes when running the app on different environments or clouds.
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

# -----------------------------------------------------------------------------
# Regex Cleaning Patterns
# -----------------------------------------------------------------------------
_RE_URL       = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION   = re.compile(r"@\w+")
_RE_HASHTAG   = re.compile(r"#\w+")
_RE_REUTERS   = re.compile(r"^.*?\(reuters\)\s*[-–]?\s*", re.IGNORECASE)
_RE_BRACKETS  = re.compile(r"\[.*?\]|\(.*?\)")
_RE_HTML      = re.compile(r"<[^>]+>|&\w+;")
_RE_NONALPHA  = re.compile(r"[^a-z\s]")
_RE_SPACES    = re.compile(r"\s+")

# -----------------------------------------------------------------------------
# Main Preprocessing Function
# -----------------------------------------------------------------------------
def preprocessing(text : str) -> str:
    if pd.isna(text):
        return ""
    
    # Text normalization & regex cleaning
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)
    text = _RE_REUTERS.sub(" ", text)
    text = _RE_BRACKETS.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_NONALPHA.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()

    # Tokenization, stopword removal, and lemmatization
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    text = " ".join(cleaned_tokens)
    
    return text