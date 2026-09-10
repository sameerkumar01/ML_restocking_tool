import pandas as pd

from src.allocation import allocate_units
from src.missing_values import missing_value_report
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
    frame = pd.DataFrame({"mobile_name": ["A"], "cost": [10000], "nation": ["India"]})
    adapter = SchemaAdapter()
    mapping = adapter.suggest_mapping(frame)
    result = adapter.transform(frame, mapping)
    assert result.loc[0, "model"] == "A"
    assert result.loc[0, "country"] == "India"


def test_schema_rejects_missing_model():
    frame = pd.DataFrame({"price": [10000]})
    adapter = SchemaAdapter()
    validation = adapter.validate(frame, adapter.suggest_mapping(frame))
    assert not validation.compatible


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


def test_knn_replaces_unavailable_product():
    frame = pd.DataFrame({
        "model": ["A", "B", "C"],
        "score": [0.9, 0.8, 0.7],
        "price_inr": [10000, 10500, 50000],
        "rating": [4.5, 4.4, 3.0],
        "battery_life_rating": [4.0, 4.1, 3.0],
        "camera_rating": [4.0, 4.1, 3.0],
        "performance_rating": [4.0, 4.1, 3.0],
        "design_rating": [4.0, 4.1, 3.0],
        "display_rating": [4.0, 4.1, 3.0],
    })
    result = select_with_substitutes(frame, unavailable=["A"], limit=2)
    assert result.loc[0, "model"] == "B"
    assert result.loc[0, "substitute_for"] == "A"
