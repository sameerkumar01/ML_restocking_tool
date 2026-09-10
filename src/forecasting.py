import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor


class DemandForecaster:
    def __init__(self, blend=0.7):
        self.blend = blend

    def forecast(self, history):
        values = pd.Series(history, dtype=float).dropna().clip(lower=0)
        if values.empty:
            return 0.0
        if len(values) < 3:
            return float(values.mean())
        exp = self._exponential(values)
        if len(values) < 8:
            return exp
        ml = self._xgboost(values)
        return max(0.0, self.blend * ml + (1 - self.blend) * exp)

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
    subset["month"] = pd.to_datetime(subset["review_date"]).dt.to_period("M")
    if "units_sold" in subset:
        return subset.groupby("month")["units_sold"].sum().sort_index()
    return subset.groupby("month").size().sort_index()
