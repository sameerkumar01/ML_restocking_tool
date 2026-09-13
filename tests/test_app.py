import sys
import types

import pandas as pd

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    streamlit_stub = types.ModuleType("streamlit")

    def identity_cache(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

    streamlit_stub.cache_data = identity_cache
    streamlit_stub.cache_resource = identity_cache
    streamlit_stub.session_state = {}
    sys.modules["streamlit"] = streamlit_stub

# Keep these pure frontend tests runnable in minimal environments. In the real
# project the genuine modules are imported because its ML dependencies exist.
try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    pipeline_stub = types.ModuleType("src.pipeline")
    pipeline_stub.InventoryPipeline = type("InventoryPipeline", (), {})
    schema_stub = types.ModuleType("src.schema")
    schema_stub.SchemaAdapter = type("SchemaAdapter", (), {})
    schema_stub.SchemaError = ValueError
    sys.modules["src.pipeline"] = pipeline_stub
    sys.modules["src.schema"] = schema_stub

from app import (
    build_effective_mapping,
    build_request_key,
    make_dataset_signature,
    parse_uploaded_data,
    prepare_plan_display,
)


def mapping_frame(rows):
    return pd.DataFrame(rows, columns=["Source", "Mapped feature"])


def test_ignored_columns_are_physically_excluded():
    frame = pd.DataFrame({"model": ["A"], "price_inr": [100], "units_sold": [50]})
    edited = mapping_frame([
        ("model", "model"),
        ("price_inr", "price_inr"),
        ("units_sold", "Ignore"),
    ])
    mapping, effective, errors = build_effective_mapping(frame, edited)
    assert errors == []
    assert mapping == {"model": "model", "price_inr": "price_inr"}
    assert list(effective.columns) == ["model", "price_inr"]


def test_duplicate_canonical_targets_are_rejected():
    frame = pd.DataFrame({"model": ["A"], "product": ["A"], "price": [100]})
    edited = mapping_frame([
        ("model", "model"),
        ("product", "model"),
        ("price", "price_inr"),
    ])
    _, effective, errors = build_effective_mapping(frame, edited)
    assert effective is None
    assert any("duplicate mappings" in error.lower() for error in errors)


def test_required_mapping_cannot_survive_as_ignored_raw_column():
    frame = pd.DataFrame({"model": ["A"], "price_inr": [100]})
    edited = mapping_frame([("model", "Ignore"), ("price_inr", "price_inr")])
    _, effective, errors = build_effective_mapping(frame, edited)
    assert effective is None
    assert any("model" in error for error in errors)


def test_dataset_signature_changes_with_mapping_metadata():
    data = pd.DataFrame({"model": ["A"], "price_inr": [100]})
    first = make_dataset_signature(data, {"source_a": "model", "source_b": "price_inr"}, "upload:x")
    second = make_dataset_signature(data, {"source_a": "brand", "source_b": "price_inr"}, "upload:x")
    assert first != second


def test_request_key_normalizes_unavailable_order():
    first = build_request_key("data", "xgboost", "India", 30, 40000, 503, ["B", "A"])
    second = build_request_key("data", "xgboost", "India", 30, 40000, 503, ["A", "B"])
    assert first == second


def test_display_score_is_scaled_without_mutating_plan():
    plan = pd.DataFrame({
        "model": ["A"],
        "price_inr": [100.0],
        "score": [0.75],
        "suggested_quantity": [1],
        "estimated_profit": [20.0],
    })
    display = prepare_plan_display(plan)
    assert display.loc[0, "Score / 100"] == 75
    assert plan.loc[0, "score"] == 0.75


def test_csv_upload_is_parsed_from_bytes():
    frame = parse_uploaded_data(b"model,price_inr\nA,100\n", "phones.csv")
    assert frame.to_dict("records") == [{"model": "A", "price_inr": 100}]
