import pandas as pd

from src.allocation import allocate_units
from src.forecasting import DemandForecaster
from src.missing_values import missing_value_report
from src.model_selection import FEATURES
from src.pipeline import InventoryPipeline
from src.schema import SchemaAdapter, SchemaError
from src.sentiment import add_sentiment_scores
from src.substitution import select_with_substitutes


def test_allocate_units_preserves_total():
    result = allocate_units([0.51, 0.31, 0.18], 101)
    assert result.sum() == 101
    assert list(result) == [52, 31, 18]


def test_allocate_units_handles_zero_weights():
    result = allocate_units([0, 0, 0], 5)
    assert result.sum() == 5


def test_schema_mapping_and_transform():
    frame = pd.DataFrame({"mobile_name": ["A"], "selling_price": [10000], "nation": ["India"]})
    adapter = SchemaAdapter()
    mapping = adapter.suggest_mapping(frame)
    result = adapter.transform(frame, mapping)
    assert result.loc[0, "model"] == "A"
    assert result.loc[0, "country"] == "India"


def test_ambiguous_cost_requires_confirmation():
    frame = pd.DataFrame({"model": ["A"], "cost": [10000]})
    mapping = SchemaAdapter().suggest_mapping(frame)
    assert "cost" not in mapping


def test_deterministic_mapping_has_priority_over_genai():
    frame = pd.DataFrame({"mobile_name": ["A"], "selling_price": [10000]})
    adapter = SchemaAdapter()
    adapter._genai_mapping = lambda *_: {"mobile_name": "brand"}
    mapping = adapter.suggest_mapping(frame, use_genai=True)
    assert mapping["mobile_name"] == "model"


def test_schema_rejects_missing_model():
    frame = pd.DataFrame({"price": [10000]})
    adapter = SchemaAdapter()
    assert not adapter.validate(frame, adapter.suggest_mapping(frame)).compatible


def test_schema_rejects_unconvertible_required_values():
    frame = pd.DataFrame({"product": ["A"], "price": ["unknown"]})
    adapter = SchemaAdapter()
    try:
        adapter.transform(frame, adapter.suggest_mapping(frame))
    except SchemaError:
        return
    raise AssertionError("Expected SchemaError")


def test_schema_recovers_inr_price_from_usd():
    frame = pd.DataFrame({"model": ["A", "B"], "price_inr": [10000, None], "price_usd": [100, 200]})
    adapter = SchemaAdapter()
    result = adapter.transform(frame, adapter.suggest_mapping(frame))
    assert result.loc[1, "price_inr"] == 17400


def test_schema_returns_rejected_rows():
    frame = pd.DataFrame({"model": ["A", ""], "price_inr": [10000, 12000]})
    adapter = SchemaAdapter()
    result = adapter.transform_with_rejections(frame, adapter.suggest_mapping(frame))
    assert len(result.data) == 1
    assert len(result.rejected) == 1
    assert result.rejected.loc[0, "rejection_reason"] == "missing model"


def test_missing_value_report_explains_strategy():
    frame = pd.DataFrame({"rating": [4.0, None], "brand": ["A", None]})
    report = missing_value_report(frame)
    assert set(report["feature"]) == {"rating", "brand"}
    assert "MICE" in report.loc[report["feature"] == "rating", "handling"].iloc[0]


def test_empty_review_text_uses_fallback_sentiment():
    frame = pd.DataFrame({"rating": [5, 2], "review_text": ["", None]})
    result, model, info = add_sentiment_scores(frame)
    assert model is None
    assert list(result["sentiment_score"]) == [1.0, 0.0]
    assert info["method"] == "label_or_rating"


def test_purchase_cost_drives_gross_profit():
    frame = pd.DataFrame({"model": ["A"], "brand": ["Apple"], "country": ["India"], "price_inr": [100.0], "purchase_cost": [70.0], "rating": [5.0]})
    result = InventoryPipeline()._prepare_raw(frame)
    assert result.loc[0, "gross_profit_unit"] == 30.0
    assert result.loc[0, "net_profit_unit"] < 30.0


def test_restock_target_is_not_a_model_feature():
    assert "restock_success" not in FEATURES


def test_recommendation_mode_uses_rule_based_fallback():
    frame = pd.DataFrame({
        "model": ["A", "B", "C"],
        "brand": ["Apple", "Samsung", "OnePlus"],
        "country": ["India", "India", "India"],
        "price_inr": [50000, 40000, 30000],
        "rating": [4.5, 4.0, 3.5],
    })
    pipeline = InventoryPipeline().fit(frame)
    result = pipeline.recommend("India", 30, 60000, 10)
    assert pipeline.metrics["ranking_mode"] == "transparent_rule_based"
    assert not result.empty
    assert result["suggested_quantity"].sum() == 10


def test_short_forecast_reports_method():
    result = DemandForecaster().forecast_with_diagnostics([5, 7])
    assert result["forecast"] == 6
    assert result["method"] == "historical_mean"


def test_knn_replaces_unavailable_product():
    frame = pd.DataFrame({
        "model": ["A", "B", "C"], "score": [0.9, 0.8, 0.7],
        "success_probability": [0.9, 0.8, 0.7], "normalized_profit": [0.8, 0.8, 0.6],
        "price_inr": [10000, 10500, 50000], "rating": [4.5, 4.4, 3.0],
        "battery_life_rating": [4.0, 4.1, 3.0], "camera_rating": [4.0, 4.1, 3.0],
        "performance_rating": [4.0, 4.1, 3.0], "design_rating": [4.0, 4.1, 3.0],
        "display_rating": [4.0, 4.1, 3.0],
    })
    result = select_with_substitutes(frame, unavailable=["A"], limit=2)
    replacement = result[result["substitute_for"] == "A"].iloc[0]
    assert replacement["model"] == "B"
    assert 0 < replacement["substitution_similarity"] <= 1
