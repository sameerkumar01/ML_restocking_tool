import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor


class DemandForecaster:
    def forecast(self, history):
        return self.forecast_with_diagnostics(history)["forecast"]

    def forecast_with_diagnostics(self, history):
        values = self._clean_history(history)
        if values.empty:
            return {"forecast": 0.0, "method": "no_history", "mae": np.nan}
        if len(values) < 3:
            return {"forecast": float(values.mean()), "method": "historical_mean", "mae": np.nan}

        validation_points = min(6, max(2, len(values) // 4))
        methods = ["naive", "exponential"]
        if len(values) - validation_points >= 18:
            methods.extend(["xgboost", "blend"])
        errors = {method: [] for method in methods}
        for index in range(len(values) - validation_points, len(values)):
            train = values.iloc[:index]
            actual = float(values.iloc[index])
            for method in methods:
                prediction = self._predict(train, method)
                errors[method].append(abs(actual - prediction))
        mean_errors = {method: float(np.mean(items)) for method, items in errors.items() if items}
        selected = min(mean_errors, key=mean_errors.get)
        return {
            "forecast": max(0.0, float(self._predict(values, selected))),
            "method": selected,
            "mae": mean_errors[selected],
        }

    def _clean_history(self, history):
        values = pd.to_numeric(pd.Series(history), errors="coerce").clip(lower=0)
        if values.isna().any():
            last_gap = np.flatnonzero(values.isna().to_numpy())[-1]
            values = values.iloc[last_gap + 1:]
        return values.dropna().reset_index(drop=True)

    def _predict(self, values, method):
        if method == "naive":
            return float(values.iloc[-1])
        if method == "xgboost":
            return self._xgboost(values)
        if method == "blend":
            return 0.7 * self._xgboost(values) + 0.3 * self._exponential(values)
        return self._exponential(values)

    def _exponential(self, values):
        try:
            fit = ExponentialSmoothing(values.to_numpy(), trend="add", seasonal=None, damped_trend=True).fit(optimized=True)
            return max(0.0, float(fit.forecast(1)[0]))
        except Exception:
            return float(values.tail(3).mean())

    def _xgboost(self, values):
        rows = []
        target = []
        data = values.to_numpy()
        for index in range(3, len(data)):
            rows.append([data[index - 1], data[index - 2], data[index - 3], np.mean(data[index - 3:index])])
            target.append(data[index])
        if len(rows) < 4:
            return float(values.tail(3).mean())
        model = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
        model.fit(np.asarray(rows), np.asarray(target))
        last = data[-3:]
        features = np.asarray([[last[-1], last[-2], last[-3], last.mean()]])
        return float(model.predict(features)[0])


def monthly_history(frame, model_name, country):
    subset = frame[(frame["model"] == model_name) & (frame["country"] == country)].copy()
    if subset.empty or "review_date" not in subset:
        return pd.Series(dtype=float)
    subset = subset.dropna(subset=["review_date"])
    if subset.empty:
        return pd.Series(dtype=float)
    subset["month"] = pd.to_datetime(subset["review_date"]).dt.to_period("M")
    full_index = pd.period_range(subset["month"].min(), subset["month"].max(), freq="M")
    if "units_sold" in subset:
        known = subset.dropna(subset=["units_sold"])
        if known.empty:
            return pd.Series(dtype=float)
        history = known.groupby("month")["units_sold"].sum().sort_index().reindex(full_index)
        return history.interpolate(limit=2, limit_area="inside")
    return subset.groupby("month").size().sort_index().reindex(full_index, fill_value=0)
