import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


POSITIVE = {"good", "great", "excellent", "love", "fast", "best", "amazing", "smooth", "reliable", "value"}
NEGATIVE = {"bad", "poor", "slow", "worst", "hate", "broken", "issue", "problem", "expensive", "return"}
NEGATIONS = {"not", "never", "no", "hardly"}
INTENSIFIERS = {"very", "extremely", "really", "highly"}


class LexiconFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = str(text).lower().split()
            size = max(len(tokens), 1)
            rows.append([
                sum(token in POSITIVE for token in tokens) / size,
                sum(token in NEGATIVE for token in tokens) / size,
                sum(token in NEGATIONS for token in tokens) / size,
                sum(token in INTENSIFIERS for token in tokens) / size,
            ])
        return csr_matrix(np.asarray(rows, dtype=float))


def build_sentiment_pipeline():
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
        ("lexicon", LexiconFeatures()),
    ])
    classifier = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    return Pipeline([("features", features), ("classifier", classifier)])
