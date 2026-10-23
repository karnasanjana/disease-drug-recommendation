"""Text cleaning shared by training and inference, so both see identical input."""
from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _pkg, _path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
_strip_punct = str.maketrans("", "", punctuation)


def clean_text(doc):
    """Lowercase, split underscores, drop punctuation/stopwords, lemmatize."""
    tokens = doc.strip().lower().replace("_", " ").split()
    tokens = [w.translate(_strip_punct) for w in tokens]
    tokens = [w for w in tokens if w.isalpha() and len(w) > 1 and w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in tokens)
