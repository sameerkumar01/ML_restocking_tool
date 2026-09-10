import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .allocation import allocate_units
from .forecasting import DemandForecaster, monthly_history


NUMERIC = ["price_inr", "age", "battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating", "sentiment_score", "net_profit_unit"]
CATEGORICAL = ["brand", "country"]


class InventoryPipeline:
    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self.model = None
        self.market = None
        self.raw = None
        self.metrics = {}
        self.forecaster = DemandForecaster()

    def fit(self, frame):
        self.raw = self._prepare_raw(frame)
        self.market = self._build_market(self.raw)
        features = CATEGORICAL + NUMERIC
        X = self.market[features]
        y = self.market["is_winner"]
        if y.nunique() < 2:
            raise ValueError("Training requires both winner and non-winner examples.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        self.model = self._model()
        self.model.fit(X_train, y_train)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        self.metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        self.model.fit(X, y)
        return self

    def recommend(self, country, age, budget, total_units, unavailable=None):
        if self.model is None:
            raise ValueError("Fit the pipeline before requesting recommendations.")
        unavailable = {str(value).lower() for value in (unavailable or [])}
        candidates = self.market[(self.market["country"] == country) & (self.market["price_inr"] <= budget)].copy()
        candidates = candidates[~candidates["model"].str.lower().isin(unavailable)]
        if candidates.empty:
            return candidates
        candidates["age"] = float(age)
        features = CATEGORICAL + NUMERIC
        candidates["success_probability"] = self.model.predict_proba(candidates[features])[:, 1]
        maximum = max(float(candidates["net_profit_unit"].max()), 1.0)
        candidates["normalized_profit"] = candidates["net_profit_unit"].clip(lower=0) / maximum
        eps = 1e-9
        candidates["score"] = 2 * candidates["success_probability"] * candidates["normalized_profit"] / (candidates["success_probability"] + candidates["normalized_profit"] + eps)
        result = candidates.sort_values("score", ascending=False).head(5).copy()
        result["forecast"] = [self.forecaster.forecast(monthly_history(self.raw, model, country)) for model in result["model"]]
        result["suggested_quantity"] = allocate_units(result["forecast"].to_numpy(), int(total_units))
        result["estimated_revenue"] = result["suggested_quantity"] * result["price_inr"]
        result["estimated_profit"] = result["suggested_quantity"] * result["net_profit_unit"]
        return result.reset_index(drop=True)

    def _prepare_raw(self, frame):
        data = frame.copy()
        defaults = {
            "brand": "Unknown",
            "country": "Global",
            "age": 30.0,
            "rating": 3.0,
            "battery_life_rating": 3.0,
            "camera_rating": 3.0,
            "performance_rating": 3.0,
            "design_rating": 3.0,
            "display_rating": 3.0,
        }
        for name, value in defaults.items():
            if name not in data:
                data[name] = value
        if "price_inr" not in data and "price_usd" in data:
            data["price_inr"] = pd.to_numeric(data["price_usd"], errors="coerce") * 87.0
        if "sentiment_score" not in data:
            if "sentiment" in data:
                data["sentiment_score"] = data["sentiment"].astype(str).str.lower().eq("positive").astype(float)
            else:
                data["sentiment_score"] = (pd.to_numeric(data["rating"], errors="coerce") >= 4).astype(float)
        margin = np.where(data["brand"].isin(["Xiaomi", "Realme", "OnePlus", "Motorola", "Vivo"]), 0.15, 0.10)
        quality = 0.7 * pd.to_numeric(data["rating"], errors="coerce").fillna(3) + 1.5 * data["sentiment_score"]
        data["return_rate"] = 0.02 + 0.18 * np.exp(-0.8 * (quality - 1))
        data["net_profit_unit"] = data["price_inr"] * margin - data["price_inr"] * data["return_rate"] * 0.4
        for name in NUMERIC:
            data[name] = pd.to_numeric(data[name], errors="coerce")
            data[name] = data[name].fillna(data[name].median() if data[name].notna().any() else 0)
        data["brand"] = data["brand"].astype(str)
        data["model"] = data["model"].astype(str)
        data["country"] = data["country"].astype(str)
        if "review_date" in data:
            data["review_date"] = pd.to_datetime(data["review_date"], errors="coerce")
        return data

    def _build_market(self, data):
        aggregations = {name: "mean" for name in NUMERIC}
        aggregations["brand"] = "first"
        if "units_sold" in data:
            aggregations["units_sold"] = "sum"
        market = data.groupby(["country", "model"], as_index=False).agg(aggregations)
        if "units_sold" in market:
            market = market.rename(columns={"units_sold": "demand"})
        else:
            counts = data.groupby(["country", "model"]).size().rename("demand").reset_index()
            market = market.merge(counts, on=["country", "model"])
        market["is_winner"] = 0
        for _, index in market.groupby("country").groups.items():
            group = market.loc[index]
            sentiment_threshold = group["sentiment_score"].quantile(0.60)
            demand_threshold = group["demand"].quantile(0.50)
            market.loc[index, "is_winner"] = ((group["sentiment_score"] >= sentiment_threshold) & (group["demand"] >= demand_threshold)).astype(int)
        return market

    def _model(self):
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", StandardScaler(), NUMERIC),
        ])
        if self.model_name == "random_forest":
            estimator = RandomForestClassifier(n_estimators=250, max_depth=10, class_weight="balanced", random_state=42)
        else:
            estimator = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
