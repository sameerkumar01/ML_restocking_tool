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
        self.mode = "recommendation"
        self.forecaster = DemandForecaster()

    def fit(self, frame):
        self.mode = self._infer_mode(frame)
        self.raw = self._prepare_raw(frame)
        self.market = self._build_market(self.raw)
        labeled = self.market.dropna(subset=["restock_success"]).copy()
        y = labeled["restock_success"].astype(int)
        if len(labeled) >= 8 and y.nunique() == 2 and y.value_counts().min() >= 2:
            X = labeled[FEATURES]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
            _, strategy, scores, holdout_auc = select_missing_strategy(self.model_name, X_train, X_test, y_train, y_test)
            self.metrics["missing_strategy_cv_roc_auc"] = scores
            self.metrics["selected_missing_strategy"] = strategy
            self.metrics["roc_auc"] = holdout_auc
            self.metrics["ranking_mode"] = "supervised"
            self.model = build_model(self.model_name, strategy)
            self.model.fit(X, y)
        else:
            self.metrics["ranking_mode"] = "transparent_rule_based"
            self.metrics["supervised_training_rows"] = int(len(labeled))
        return self

    def recommend(self, country, age, budget, total_units, unavailable=None):
        candidates = self.market[(self.market["country"] == country) & (self.market["price_inr"] <= budget) & (self.market["net_profit_unit"] > 0)].copy()
        if candidates.empty:
            return candidates
        candidates["age"] = float(age)
        if self.model is not None:
            candidates["success_probability"] = self.model.predict_proba(candidates[FEATURES])[:, 1]
        else:
            candidates["success_probability"] = self._rule_based_probability(candidates)
        maximum = max(float(candidates["net_profit_unit"].max()), 1.0)
        candidates["normalized_profit"] = candidates["net_profit_unit"].clip(lower=0) / maximum
        eps = 1e-9
        candidates["score"] = 2 * candidates["success_probability"] * candidates["normalized_profit"] / (candidates["success_probability"] + candidates["normalized_profit"] + eps)
        result = select_with_substitutes(candidates, unavailable, limit=5)
        if result.empty:
            return result
        forecasts = [self.forecaster.forecast_with_diagnostics(monthly_history(self.raw, model, country)) for model in result["model"]]
        result["forecast"] = [item["forecast"] for item in forecasts]
        result["forecast_method"] = [item["method"] for item in forecasts]
        result["forecast_mae"] = [item["mae"] for item in forecasts]
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
        for name in RATING_NUMERIC + ["age", "purchase_cost", "units_sold", "restock_success"]:
            if name in data:
                data[name] = pd.to_numeric(data[name], errors="coerce")
        for name in RATING_NUMERIC:
            if name not in data:
                data[name] = np.nan
            data[f"{name}_missing_rate"] = data[name].isna().astype(float)
        if "age" not in data:
            data["age"] = np.nan
        country_age = data.groupby("country")["age"].transform("median")
        global_age = data["age"].median() if data["age"].notna().any() else 30.0
        data["age"] = data["age"].fillna(country_age).fillna(global_age)
        if "purchase_cost" not in data:
            data["purchase_cost"] = np.nan
        invalid_cost = data["purchase_cost"].le(0) | data["purchase_cost"].gt(data["price_inr"] * 3)
        data.loc[invalid_cost, "purchase_cost"] = np.nan
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
        aggregations.update({f"{name}_missing_rate": "mean" for name in RATING_NUMERIC})
        aggregations.update({"brand": "first", "purchase_cost": "mean"})
        if "restock_success" in data:
            aggregations["restock_success"] = "mean"
        market = data.groupby(["country", "model"], as_index=False).agg(aggregations)
        if "restock_success" in market:
            market["restock_success"] = market["restock_success"].where(market["restock_success"].isna(), (market["restock_success"] >= 0.5).astype(float))
        else:
            market["restock_success"] = np.nan
        if "units_sold" in data:
            known = data.dropna(subset=["units_sold"])
            demand = known.groupby(["country", "model"], as_index=False)["units_sold"].sum().rename(columns={"units_sold": "demand"})
            market = market.merge(demand, on=["country", "model"], how="left")
        else:
            counts = data.groupby(["country", "model"]).size().rename("demand").reset_index()
            market = market.merge(counts, on=["country", "model"])
        market["demand"] = market["demand"].fillna(0)
        return market

    def _rule_based_probability(self, candidates):
        rating = candidates[RATING_NUMERIC].mean(axis=1, skipna=True).fillna(2.5).clip(0, 5) / 5
        sentiment = candidates["sentiment_score"].fillna(0.5).clip(0, 1)
        demand = np.log1p(candidates["demand"].clip(lower=0))
        demand = demand / max(float(demand.max()), 1.0)
        return (0.45 * sentiment + 0.35 * rating + 0.20 * demand).clip(0, 1)

    def _infer_mode(self, frame):
        has_time = "review_date" in frame
        has_demand = "units_sold" in frame or "review_id" in frame
        has_reviews = "review_text" in frame or "sentiment" in frame
        if has_time and has_demand:
            return "full" if has_reviews else "forecast"
        if has_reviews:
            return "review"
        return "recommendation"
