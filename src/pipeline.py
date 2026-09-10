import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .allocation import allocate_units
from .forecasting import DemandForecaster, monthly_history
from .model_selection import CATEGORICAL, FEATURES, NUMERIC, RATING_NUMERIC, build_model, select_missing_strategy
from .sentiment import add_sentiment_scores
from .substitution import select_with_substitutes


class InventoryPipeline:
    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self.model = None
        self.sentiment_model = None
        self.market = None
        self.raw = None
        self.metrics = {}
        self.forecaster = DemandForecaster()

    def fit(self, frame):
        self.raw = self._prepare_raw(frame)
        self.market = self._build_market(self.raw)
        X = self.market[FEATURES]
        y = self.market["is_winner"]
        if y.nunique() < 2:
            raise ValueError("Training requires both winner and non-winner examples.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        _, strategy, scores = select_missing_strategy(self.model_name, X_train, X_test, y_train, y_test)
        self.metrics["missing_strategy_roc_auc"] = scores
        self.metrics["selected_missing_strategy"] = strategy
        self.metrics["roc_auc"] = scores[strategy]
        self.model = build_model(self.model_name, strategy)
        self.model.fit(X, y)
        return self

    def recommend(self, country, age, budget, total_units, unavailable=None):
        if self.model is None:
            raise ValueError("Fit the pipeline before requesting recommendations.")
        candidates = self.market[(self.market["country"] == country) & (self.market["price_inr"] <= budget)].copy()
        if candidates.empty:
            return candidates
        candidates["age"] = float(age)
        candidates["success_probability"] = self.model.predict_proba(candidates[FEATURES])[:, 1]
        maximum = max(float(candidates["net_profit_unit"].max()), 1.0)
        candidates["normalized_profit"] = candidates["net_profit_unit"].clip(lower=0) / maximum
        eps = 1e-9
        candidates["score"] = 2 * candidates["success_probability"] * candidates["normalized_profit"] / (candidates["success_probability"] + candidates["normalized_profit"] + eps)
        result = select_with_substitutes(candidates, unavailable, limit=5)
        if result.empty:
            return result
        result["forecast"] = [self.forecaster.forecast(monthly_history(self.raw, model, country)) for model in result["model"]]
        result["suggested_quantity"] = allocate_units(result["forecast"].to_numpy(), int(total_units))
        result["estimated_revenue"] = result["suggested_quantity"] * result["price_inr"]
        result["estimated_profit"] = result["suggested_quantity"] * result["net_profit_unit"]
        return result.reset_index(drop=True)

    def _prepare_raw(self, frame):
        data = frame.copy()
        for name in CATEGORICAL:
            if name not in data:
                data[name] = "Unknown" if name == "brand" else "Global"
            data[name] = data[name].astype("string").str.strip().replace("", pd.NA).fillna("Unknown" if name == "brand" else "Global")
        data["model"] = data["model"].astype("string").str.strip()
        for name in RATING_NUMERIC + ["age", "purchase_cost", "units_sold"]:
            if name in data:
                data[name] = pd.to_numeric(data[name], errors="coerce")
        for name in RATING_NUMERIC:
            if name not in data:
                data[name] = np.nan
        if "age" not in data:
            data["age"] = np.nan
        country_age = data.groupby("country")["age"].transform("median")
        global_age = data["age"].median() if data["age"].notna().any() else 30.0
        data["age"] = data["age"].fillna(country_age).fillna(global_age)
        if "purchase_cost" not in data:
            data["purchase_cost"] = np.nan
        brand_cost = data.groupby("brand")["purchase_cost"].transform("median")
        global_cost = data["purchase_cost"].median()
        data["purchase_cost"] = data["purchase_cost"].fillna(brand_cost).fillna(global_cost)
        if "review_text" in data:
            data["review_text"] = data["review_text"].fillna("").astype(str)
        data, self.sentiment_model, sentiment_info = add_sentiment_scores(data)
        self.metrics["sentiment"] = sentiment_info
        rating_median = data.groupby("brand")["rating"].transform("median")
        global_rating = data["rating"].median() if data["rating"].notna().any() else 3.0
        quality_rating = data["rating"].fillna(rating_median).fillna(global_rating)
        margin = np.where(data["brand"].isin(["Xiaomi", "Realme", "OnePlus", "Motorola", "Vivo"]), 0.15, 0.10)
        fallback_cost = data["price_inr"] * (1 - margin)
        data["purchase_cost"] = data["purchase_cost"].fillna(fallback_cost)
        quality = 0.7 * quality_rating + 1.5 * data["sentiment_score"]
        data["return_rate"] = 0.02 + 0.18 * np.exp(-0.8 * (quality - 1))
        data["gross_profit_unit"] = data["price_inr"] - data["purchase_cost"]
        data["net_profit_unit"] = data["gross_profit_unit"] - data["price_inr"] * data["return_rate"] * 0.4
        if "review_date" in data:
            data["review_date"] = pd.to_datetime(data["review_date"], errors="coerce")
        return data

    def _build_market(self, data):
        aggregations = {name: "mean" for name in NUMERIC}
        aggregations.update({"brand": "first", "purchase_cost": "mean"})
        market = data.groupby(["country", "model"], as_index=False).agg(aggregations)
        if "units_sold" in data:
            known = data.dropna(subset=["units_sold"])
            demand = known.groupby(["country", "model"], as_index=False)["units_sold"].sum().rename(columns={"units_sold": "demand"})
            market = market.merge(demand, on=["country", "model"], how="inner")
            if market.empty:
                raise ValueError("Training requires known units_sold values; missing regression targets are not imputed.")
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
