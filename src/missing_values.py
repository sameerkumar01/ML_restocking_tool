import pandas as pd


STRATEGIES = {
    "model": "Reject missing or blank identifiers",
    "price_inr": "Recover from USD or reject",
    "price_usd": "Use INR when available",
    "age": "Country median, then global median",
    "purchase_cost": "Brand median, global median, then margin fallback",
    "brand": "Fill with Unknown",
    "country": "Fill with Global",
    "review_text": "Fill with an empty string",
    "units_sold": "Keep missing; never impute a training target",
    "review_date": "Reject invalid forecasting rows",
}


def strategy_for(column):
    if column.endswith("_rating") or column == "rating":
        return "MICE imputation inside the training pipeline"
    return STRATEGIES.get(column, "Median imputation for numeric model features")


def missing_value_report(frame):
    rows = []
    total = max(len(frame), 1)
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        if missing:
            rows.append({
                "feature": column,
                "missing_values": missing,
                "missing_percent": round(100 * missing / total, 2),
                "handling": strategy_for(column),
            })
    return pd.DataFrame(rows, columns=["feature", "missing_values", "missing_percent", "handling"])
