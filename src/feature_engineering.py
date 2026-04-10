from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    """Get TF-IDF vectorizer for text feature extraction."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )