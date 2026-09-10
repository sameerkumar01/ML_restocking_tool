from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RATING_NUMERIC = ["rating", "battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]
OTHER_NUMERIC = ["price_inr", "age", "sentiment_score", "net_profit_unit"]
CATEGORICAL = ["brand", "country"]
FEATURES = CATEGORICAL + RATING_NUMERIC + OTHER_NUMERIC


def build_model(model_name="xgboost", missing_strategy="mice"):
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    if missing_strategy == "native":
        preprocessor = ColumnTransformer([
            ("categorical", categorical, CATEGORICAL),
            ("numeric", "passthrough", RATING_NUMERIC + OTHER_NUMERIC),
        ])
    else:
        ratings = Pipeline([
            ("imputer", IterativeImputer(initial_strategy="median", max_iter=10, random_state=42, skip_complete=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        preprocessor = ColumnTransformer([
            ("categorical", categorical, CATEGORICAL),
            ("ratings_mice", ratings, RATING_NUMERIC),
            ("numeric", numeric, OTHER_NUMERIC),
        ])
    if model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=250, max_depth=10, class_weight="balanced", random_state=42)
    else:
        estimator = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def select_missing_strategy(model_name, X_train, X_test, y_train, y_test):
    strategies = ["mice", "native"] if model_name == "xgboost" else ["mice"]
    scores = {}
    models = {}
    for strategy in strategies:
        model = build_model(model_name, strategy)
        model.fit(X_train, y_train)
        scores[strategy] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        models[strategy] = model
    selected = max(scores, key=scores.get)
    return models[selected], selected, scores
