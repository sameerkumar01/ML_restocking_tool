import numpy as np
import pandas as pd
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


def add_sentiment_scores(frame):
    data = frame.copy()
    rating = pd.to_numeric(data.get("rating", pd.Series(index=data.index, dtype=float)), errors="coerce")
    baseline = (rating >= 4).astype(float)
    if "sentiment" in data:
        labels = data["sentiment"].astype("string").str.lower()
        baseline = labels.map({"positive": 1.0, "negative": 0.0}).fillna(baseline)
    data["sentiment_score"] = baseline.fillna(0.5)
    if "review_text" not in data:
        return data, None, {"method": "label_or_rating", "training_rows": 0}

    text = data["review_text"].fillna("").astype(str).str.strip()
    labels = data.get("sentiment", pd.Series(index=data.index, dtype="string")).astype("string").str.lower()
    train_mask = text.ne("") & labels.isin(["positive", "negative"])
    counts = labels.loc[train_mask].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        return data, None, {"method": "label_or_rating", "training_rows": int(train_mask.sum())}

    model = build_sentiment_pipeline()
    target = labels.loc[train_mask].eq("positive").astype(int)
    model.fit(text.loc[train_mask], target)
    score_mask = text.ne("")
    probabilities = model.predict_proba(text.loc[score_mask])
    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    data.loc[score_mask, "sentiment_score"] = probabilities[:, positive_index]
    return data, model, {"method": "tfidf_lexicon_linear_svm", "training_rows": int(train_mask.sum())}
