import pandas as pd

from src.allocation import allocate_units
from src.schema import SchemaAdapter, SchemaError


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
    validation = adapter.validate(frame, mapping)
    result = adapter.transform(frame, mapping)
    assert validation.compatible
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
