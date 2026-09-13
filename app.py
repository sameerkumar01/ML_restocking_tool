import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.missing_values import missing_value_report
from src.pipeline import InventoryPipeline
from src.schema import SchemaAdapter, SchemaError


APP_DIR = Path(__file__).resolve().parent
BUILTIN_NAME = "Mobile Reviews Sentiment.csv"
BUILTIN_PATH = APP_DIR / BUILTIN_NAME
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 500_000
MAX_COLUMNS = 200
CANONICAL_OPTIONS = [
    "Ignore",
    "review_id",
    "brand",
    "model",
    "price_usd",
    "price_inr",
    "country",
    "age",
    "review_date",
    "review_text",
    "sentiment",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "units_sold",
    "purchase_cost",
    "restock_success",
]
ALLOWED_TARGETS = set(CANONICAL_OPTIONS) - {"Ignore"}
REPOSITORY_URL = "https://github.com/sameerkumar01/ML_restocking_tool"


@st.cache_data(show_spinner=False)
def load_builtin(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def train_pipeline(data_fingerprint, model_name, _data):
    return InventoryPipeline(model_name=model_name).fit(_data)


def fingerprint(frame):
    values = pd.util.hash_pandas_object(frame.astype(str), index=True).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_dataset_signature(data, mapping, source_id):
    metadata = json.dumps(
        {
            "source": source_id,
            "columns": [str(column) for column in data.columns],
            "dtypes": [str(dtype) for dtype in data.dtypes],
            "mapping": sorted((str(key), str(value)) for key, value in mapping.items()),
        },
        sort_keys=True,
    )
    payload = metadata + "\n" + fingerprint(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_request_key(dataset_signature, model_name, country, age, budget, total_units, unavailable):
    return (
        dataset_signature,
        model_name,
        country,
        int(age),
        float(budget),
        int(total_units),
        tuple(sorted(str(item) for item in unavailable)),
    )


def parse_uploaded_data(payload, filename):
    if not payload:
        raise ValueError("The selected file is empty.")
    suffix = Path(filename).suffix.lower()
    stream = io.BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(stream)
    if suffix == ".xlsx":
        return pd.read_excel(stream)
    raise ValueError("Choose a CSV or XLSX file.")


def build_effective_mapping(frame, edited_mapping):
    errors = []
    if frame.columns.duplicated().any():
        duplicates = sorted({str(value) for value in frame.columns[frame.columns.duplicated()]})
        return {}, None, ["Duplicate source column names are not supported: " + ", ".join(duplicates)]

    mapping = {}
    targets = []
    source_names = set(frame.columns)
    for _, row in edited_mapping.iterrows():
        source = row.get("Source")
        target = row.get("Mapped feature")
        if target == "Ignore" or pd.isna(target):
            continue
        if source not in source_names:
            errors.append(f"Mapped source column does not exist: {source}")
            continue
        if target not in ALLOWED_TARGETS:
            errors.append(f"Unsupported mapped feature: {target}")
            continue
        mapping[source] = target
        targets.append(target)

    duplicate_targets = sorted({target for target in targets if targets.count(target) > 1})
    if duplicate_targets:
        errors.append(
            "Each app field can be mapped only once. Resolve duplicate mappings for: "
            + ", ".join(duplicate_targets)
        )
    if not mapping:
        errors.append("Map at least a product/model column and a selling-price column.")
    if "model" not in mapping.values():
        errors.append("Map one source column to model.")
    if not ({"price_inr", "price_usd"} & set(mapping.values())):
        errors.append("Map one source column to price_inr or price_usd.")
    if errors:
        return mapping, None, errors

    effective_frame = frame.loc[:, list(mapping.keys())].copy()
    renamed = [mapping[column] for column in effective_frame.columns]
    if len(renamed) != len(set(renamed)):
        return mapping, None, ["The selected mappings would create duplicate columns."]
    return mapping, effective_frame, []


def prepare_plan_display(plan):
    required = ["model", "price_inr", "score", "suggested_quantity", "estimated_profit"]
    if any(column not in plan.columns for column in required):
        return pd.DataFrame()
    display = plan[required].copy()
    display["score"] = pd.to_numeric(display["score"], errors="coerce") * 100
    return display.rename(
        columns={
            "model": "Phone model",
            "price_inr": "Unit price",
            "score": "Score / 100",
            "suggested_quantity": "Units to stock",
            "estimated_profit": "Est. net profit",
        }
    )


def safe_number(value, decimals=2):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    return f"{number:,.{decimals}f}"


def money(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    return f"₹{number:,.0f}"


def source_note(data):
    if "units_sold" not in data.columns:
        return "Demand forecasts use monthly record counts as a proxy because units_sold is not available."
    sales = pd.to_numeric(data["units_sold"], errors="coerce")
    if sales.notna().sum() == 0:
        return "A units_sold column is present, but it contains no usable sales values; forecasts may report no history."
    if sales.isna().any():
        return "Demand forecasts use known units_sold values; missing sales remain excluded and results may be incomplete."
    return "Demand forecasts use the recorded units_sold history."


def mode_label(mode):
    return {
        "full": "Full",
        "forecast": "Forecast",
        "review": "Review",
        "recommendation": "Recommendation",
    }.get(mode, str(mode).replace("_", " ").title())


def empty_dataset_context(source_name="No dataset selected", errors=None):
    return {
        "ready": False,
        "source_name": source_name,
        "errors": errors or [],
        "frame": None,
        "data": None,
        "mapping": {},
        "signature": None,
        "mode": None,
        "rejected": pd.DataFrame(),
        "source_note": "Demand source is not available until a dataset is ready.",
    }


def get_suggestions(adapter, frame, source_id, use_genai):
    cache = st.session_state.setdefault("mapping_suggestions", {})
    cache_key = f"{source_id}:{'gemini' if use_genai else 'deterministic'}"
    if cache_key not in cache:
        error = None
        try:
            suggestion = adapter.suggest_mapping(frame, use_genai=use_genai)
        except SchemaError as exc:
            error = str(exc)
            suggestion = adapter.suggest_mapping(frame, use_genai=False)
        cache[cache_key] = {"mapping": suggestion, "error": error}
    return cache[cache_key]


def render_dataset_tab():
    st.header("Choose your dataset")
    st.caption("Use the bundled mobile reviews file or upload a CSV/XLSX dataset.")

    source_choice = st.radio(
        "Data source",
        ["Built-in dataset", "Upload a file"],
        horizontal=True,
        key="data_source_choice",
    )

    frame = None
    source_name = BUILTIN_NAME
    source_id = f"builtin:{BUILTIN_PATH}"
    use_genai = False

    if source_choice == "Built-in dataset":
        if not BUILTIN_PATH.exists():
            st.error(f"The built-in dataset was not found at {BUILTIN_PATH}.")
            return empty_dataset_context(BUILTIN_NAME, ["Built-in dataset is missing."])
        try:
            frame = load_builtin(str(BUILTIN_PATH)).copy()
        except Exception as exc:
            st.error(f"The built-in dataset could not be read: {exc}")
            return empty_dataset_context(BUILTIN_NAME, [str(exc)])
    else:
        uploaded = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"], key="dataset_upload")
        if uploaded is None:
            st.info("Choose a file to prepare the dataset.")
            return empty_dataset_context("Waiting for an upload")
        source_name = uploaded.name
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("Uploads are limited to 50 MB.")
            return empty_dataset_context(source_name, ["File exceeds 50 MB."])
        payload = uploaded.getvalue()
        source_id = f"upload:{uploaded.name}:{hashlib.sha256(payload).hexdigest()}"
        try:
            frame = parse_uploaded_data(payload, uploaded.name)
        except Exception as exc:
            st.error(f"The file could not be read. Check its format and encoding. Details: {exc}")
            return empty_dataset_context(source_name, [str(exc)])
        use_genai = st.checkbox(
            "Use Gemini for column-mapping suggestions",
            value=False,
            key=f"use_gemini:{source_id}",
            help="Gemini assists column mapping only; it does not create recommendations.",
        )
        st.caption(
            "If enabled, column names, data types, and whether columns contain missing values are sent to Google Gemini. "
            "Row values and review text are not included."
        )

    if frame is None:
        return empty_dataset_context(source_name)
    if frame.empty or len(frame.columns) == 0:
        st.error("The dataset is empty. Add at least one row and the required columns.")
        return empty_dataset_context(source_name, ["Dataset is empty."])
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        st.error("Uploads are limited to 500,000 rows and 200 columns.")
        return empty_dataset_context(source_name, ["Dataset exceeds the supported dimensions."])
    if frame.columns.duplicated().any():
        duplicates = sorted({str(value) for value in frame.columns[frame.columns.duplicated()]})
        st.error("Duplicate source column names are not supported: " + ", ".join(duplicates))
        return empty_dataset_context(source_name, ["Duplicate source columns."])

    st.write(f"**{source_name}**")
    st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns")

    adapter = SchemaAdapter()
    suggestion_bundle = get_suggestions(adapter, frame, source_id, use_genai)
    if suggestion_bundle["error"]:
        st.warning(
            "Gemini mapping was unavailable, so deterministic suggestions are shown. "
            + suggestion_bundle["error"]
        )
    suggestions = suggestion_bundle["mapping"]

    reset_state_key = f"mapping_reset_revision:{source_id}:{use_genai}"
    if reset_state_key not in st.session_state:
        st.session_state[reset_state_key] = 0

    with st.expander("Review column mapping", expanded=False):
        if st.button("Reset mapping", key=f"reset_mapping_button:{source_id}:{use_genai}"):
            st.session_state[reset_state_key] += 1
        revision = st.session_state[reset_state_key]
        editor_key = f"mapping_editor:{source_id}:{use_genai}:{revision}"
        editable = pd.DataFrame(
            [
                {"Source": column, "Mapped feature": suggestions.get(column, "Ignore")}
                for column in frame.columns
            ]
        )
        edited = st.data_editor(
            editable,
            hide_index=True,
            disabled=["Source"],
            column_config={
                "Mapped feature": st.column_config.SelectboxColumn(
                    "Mapped feature", options=CANONICAL_OPTIONS, required=True
                )
            },
            use_container_width=True,
            key=editor_key,
        )
        st.caption("Ignore excludes a source column from validation, training, and recommendations.")

    mapping, effective_frame, mapping_errors = build_effective_mapping(frame, edited)
    if mapping_errors:
        for error in mapping_errors:
            st.error(error)
        return empty_dataset_context(source_name, mapping_errors)

    validation = adapter.validate(effective_frame, mapping)
    if validation.errors:
        for error in validation.errors:
            st.error(error)
        return empty_dataset_context(source_name, list(validation.errors))
    for warning in validation.warnings:
        st.warning(warning)

    try:
        transformed = adapter.transform_with_rejections(effective_frame, mapping)
    except SchemaError as exc:
        st.error(str(exc))
        return empty_dataset_context(source_name, [str(exc)])

    data = transformed.data
    rejected = transformed.rejected
    signature = make_dataset_signature(data, mapping, source_id)
    note = source_note(data)

    st.success(f"Dataset ready in {mode_label(validation.mode)} mode.")
    st.caption(
        f"{len(data):,} valid rows · {len(rejected):,} rejected rows · {note} "
        "USD prices are converted to INR with the pipeline's fixed factor of 87.0."
    )

    report = missing_value_report(data)
    with st.expander("Validation and missing-value handling", expanded=False):
        if rejected.empty:
            st.write("No rows were rejected by required-field validation.")
        else:
            st.warning(f"{len(rejected):,} rows were rejected; {len(data):,} valid rows remain.")
            st.dataframe(rejected.head(100), use_container_width=True, hide_index=True)
        if report.empty:
            st.write("No missing values remain in the canonical valid rows.")
        else:
            st.dataframe(report, use_container_width=True, hide_index=True)
            st.caption(
                "These are handling policies. Model preprocessing is fitted inside training folds; "
                "the selected model may use native missing-value handling instead."
            )

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download canonical valid rows",
        data.to_csv(index=False),
        "canonical_valid_rows.csv",
        "text/csv",
        key=f"download_valid:{signature}",
    )
    if not rejected.empty:
        download_right.download_button(
            "Download rejected rows",
            rejected.to_csv(index=False),
            "rejected_rows.csv",
            "text/csv",
            key=f"download_rejected:{signature}",
        )

    with st.expander("Preview current source data", expanded=False):
        st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
        st.caption("source_row in validation exports refers to the original DataFrame index, not an exact CSV line.")

    return {
        "ready": True,
        "source_name": source_name,
        "errors": [],
        "frame": frame,
        "data": data,
        "mapping": mapping,
        "signature": signature,
        "mode": validation.mode,
        "rejected": rejected,
        "source_note": note,
    }


def render_diagnostics(metrics, mode, source_context):
    ranking_mode = metrics.get("ranking_mode")
    if ranking_mode == "supervised":
        st.write("**Ranking:** Supervised product-success model")
        st.write(f"Selected missing-value strategy: `{metrics.get('selected_missing_strategy', 'Not evaluated')}`")
        comparisons = metrics.get("missing_strategy_cv_roc_auc")
        if comparisons:
            st.write("Preprocessing comparison ROC-AUC:", comparisons)
        st.write("Ranking holdout ROC-AUC:", safe_number(metrics.get("roc_auc"), 3))
    else:
        st.write("**Ranking:** Transparent rule-based ranking")
        rows = metrics.get("supervised_training_rows")
        if rows is not None:
            st.write(f"Observed supervised training rows: {rows}")

    sentiment = metrics.get("sentiment") or {}
    st.write(f"**Sentiment method:** {sentiment.get('method', 'Not evaluated')}")
    st.write(f"Sentiment training rows: {sentiment.get('training_rows', 'Not evaluated')}")
    if sentiment.get("oof_roc_auc") is not None:
        st.write("Sentiment out-of-fold ROC-AUC:", safe_number(sentiment.get("oof_roc_auc"), 3))
        st.write("Sentiment out-of-fold F1:", safe_number(sentiment.get("oof_f1"), 3))
    if sentiment.get("fallback_reason"):
        st.caption(f"Sentiment fallback: {sentiment['fallback_reason']}")
    st.caption(f"Dataset mode: {mode_label(mode)}. {source_context}")


def render_phone_details(visible_plan, result_id, ranking_mode):
    with st.expander("Phone details", expanded=False):
        if visible_plan.empty:
            st.info("No visible phone is available for details.")
            return
        options = list(visible_plan.index)
        selection_key = f"phone_detail:{result_id}"
        if st.session_state.get(selection_key) not in options:
            st.session_state[selection_key] = options[0]
        selected_index = st.selectbox(
            "Select a phone",
            options,
            key=selection_key,
            format_func=lambda index: f"{visible_plan.loc[index].get('brand', '')} {visible_plan.loc[index].get('model', '')}".strip(),
        )
        row = visible_plan.loc[selected_index]
        st.write(f"**{row.get('brand', '')} {row.get('model', '')}**".strip())
        detail_rows = [
            ("Average selling price", money(row.get("price_inr"))),
            ("Purchase cost", money(row.get("purchase_cost"))),
            ("Estimated net profit per unit", money(row.get("net_profit_unit"))),
            ("Assigned units", safe_number(row.get("suggested_quantity"), 0)),
            ("Estimated revenue", money(row.get("estimated_revenue"))),
            ("Estimated net profit", money(row.get("estimated_profit"))),
            ("Ranking score", safe_number(float(row.get("score")) * 100 if pd.notna(row.get("score")) else None, 1) + " / 100" if pd.notna(row.get("score")) else "Not available"),
            (
                "Estimated success probability" if ranking_mode == "supervised" else "Rule-based strength",
                safe_number(float(row.get("success_probability")) * 100 if pd.notna(row.get("success_probability")) else None, 1) + "%" if pd.notna(row.get("success_probability")) else "Not available",
            ),
            ("Demand forecast", safe_number(row.get("forecast"), 2)),
            ("Forecast method", row.get("forecast_method", "Not available")),
            ("Forecast validation MAE", safe_number(row.get("forecast_mae"), 2)),
        ]
        substitute_for = row.get("substitute_for")
        if pd.notna(substitute_for) and str(substitute_for).strip():
            detail_rows.extend(
                [
                    ("Substitute for", str(substitute_for)),
                    ("Substitution similarity", safe_number(float(row.get("substitution_similarity")) * 100, 1) + "%"),
                ]
            )
        details = pd.DataFrame(detail_rows, columns=["Field", "Value"])
        st.dataframe(details, use_container_width=True, hide_index=True)


def render_results(bundle, dataset_context):
    plan = bundle["plan"].copy()
    metrics = bundle["metrics"]
    ranking_mode = metrics.get("ranking_mode")
    result_id = hashlib.sha256(repr(bundle["key"]).encode("utf-8")).hexdigest()[:12]

    heading_col, download_col = st.columns([3, 1])
    heading_col.header("2. Your results")
    download_col.download_button(
        "Download CSV",
        plan.to_csv(index=False),
        "stocking_plan.csv",
        "text/csv",
        key=f"download_plan:{result_id}",
        use_container_width=True,
    )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Total stock", f"{int(plan['suggested_quantity'].sum()):,}")
    metric_two.metric("Estimated revenue", money(plan["estimated_revenue"].sum()))
    metric_three.metric("Estimated net profit", money(plan["estimated_profit"].sum()))
    st.caption(
        "Revenue assumes assigned units sell at their listed average prices. Profit includes the pipeline's purchase-cost and return estimates."
    )

    st.subheader("Recommended phones")
    search = st.text_input("Search by model or brand", key=f"plan_search:{result_id}", placeholder="Search phones...")
    visible_plan = plan
    if search.strip():
        query = search.strip()
        model_match = plan.get("model", pd.Series(index=plan.index, dtype=str)).astype("string").str.contains(
            query, case=False, regex=False, na=False
        )
        brand_match = plan.get("brand", pd.Series(index=plan.index, dtype=str)).astype("string").str.contains(
            query, case=False, regex=False, na=False
        )
        visible_plan = plan.loc[model_match | brand_match]

    if visible_plan.empty:
        st.info("No phones match your search. Try a different model or brand name.")
    else:
        display = prepare_plan_display(visible_plan)
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Unit price": st.column_config.NumberColumn(format="₹%.0f"),
                "Score / 100": st.column_config.NumberColumn(format="%.1f"),
                "Units to stock": st.column_config.NumberColumn(format="%d"),
                "Est. net profit": st.column_config.NumberColumn(format="₹%.0f"),
            },
        )
    st.caption(
        f"{len(visible_plan)} of {len(plan)} phones shown · {bundle['context']['country']} · "
        "Scores are rankings, not success probabilities."
    )

    render_phone_details(visible_plan, result_id, ranking_mode)

    st.subheader("Stock distribution")
    measure = st.selectbox(
        "Chart measure",
        ["Units to stock", "Demand forecast"],
        key=f"chart_measure:{result_id}",
    )
    selected_field = "suggested_quantity" if measure == "Units to stock" else "forecast"
    if selected_field in plan.columns:
        chart_data = plan[["model", selected_field]].rename(columns={selected_field: "value"}).copy()
        chart_data["value"] = pd.to_numeric(chart_data["value"], errors="coerce").fillna(0)
        spec = {
            "mark": {"type": "bar", "color": "#496d9f"},
            "encoding": {
                "x": {"field": "value", "type": "quantitative", "title": measure, "scale": {"zero": True}},
                "y": {"field": "model", "type": "nominal", "title": None, "sort": "-x"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "title": "Phone"},
                    {"field": "value", "type": "quantitative", "title": measure},
                ],
            },
            "height": {"step": 28},
        }
        st.vega_lite_chart(chart_data, spec, use_container_width=True)
    st.caption(dataset_context["source_note"] if measure == "Demand forecast" else "Whole-unit allocations add up to the requested stock total.")
    if "forecast" in plan.columns and pd.to_numeric(plan["forecast"], errors="coerce").fillna(0).eq(0).all():
        st.info(
            "All selected forecast weights are zero. The existing allocation rule therefore distributes units using equal weights and integer rounding."
        )

    with st.expander("Why were these phones selected?", expanded=False):
        if ranking_mode == "supervised":
            st.write(
                "A supervised model estimated the probability of the observed restock_success outcome. "
                "The final ranking then balanced that estimate with normalized expected unit profit."
            )
        else:
            st.write(
                "A transparent score combined sentiment (45%), average ratings (35%), and normalized demand activity (20%). "
                "The final ranking balanced that score with normalized expected unit profit."
            )
        st.write(
            "Products above the selected unit-price limit or with non-positive estimated unit profit were excluded. "
            "Unavailable products could be replaced by similar profitable alternatives."
        )
        render_diagnostics(metrics, bundle["mode"], dataset_context["source_note"])


def render_recommendations_tab(dataset_context):
    st.header("1. Enter your preferences")
    if not dataset_context["ready"]:
        st.caption("Open the Dataset tab above to choose and validate your data.")
        st.info("A valid dataset is required before recommendations can be generated.")
        st.session_state.pop("restock_result", None)
        st.divider()
        st.header("2. Your results")
        st.info("Choose your preferences and generate recommendations.")
        return

    data = dataset_context["data"]
    signature = dataset_context["signature"]
    widget_scope = signature[:12]
    st.caption(
        f"Using {dataset_context['source_name']} · {mode_label(dataset_context['mode'])} mode. "
        "Open the Dataset tab above to change your data."
    )

    normalized_country = data["country"].astype("string").str.strip().replace("", pd.NA).fillna("Global")
    countries = sorted(normalized_country.unique().tolist())
    model_names = sorted(
        data["model"].astype("string").str.strip().dropna().loc[lambda values: values.ne("")].unique().tolist()
    )
    median_price = pd.to_numeric(data["price_inr"], errors="coerce").median()
    budget_default = float(median_price) if pd.notna(median_price) and math.isfinite(float(median_price)) else 1.0
    budget_default = max(1.0, budget_default)

    col_country, col_age, col_budget, col_units = st.columns(4)
    country = col_country.selectbox(
        "Country",
        countries,
        key=f"country:{widget_scope}",
        help="Global is the fallback country bucket, not all markets combined.",
    )
    age = col_age.number_input(
        "Customer age",
        min_value=13,
        max_value=100,
        value=30,
        step=1,
        key=f"age:{widget_scope}",
        help="Used as a model input. It does not change rule-based rankings.",
    )
    budget = col_budget.number_input(
        "Max. phone price (INR)",
        min_value=1.0,
        value=budget_default,
        key=f"budget:{widget_scope}",
        help="Maximum average selling price of one phone, in INR.",
    )
    total_units = col_units.number_input(
        "Total units to stock",
        min_value=1,
        value=500,
        step=1,
        key=f"units:{widget_scope}",
    )

    with st.expander("More settings (optional)", expanded=False):
        settings_left, settings_right = st.columns([1, 2])
        model_name = settings_left.selectbox(
            "ML model",
            ["xgboost", "random_forest"],
            format_func=lambda value: "XGBoost" if value == "xgboost" else "Random Forest",
            key=f"model_name:{widget_scope}",
            help="Used only when enough observed restock_success labels are available.",
        )
        unavailable_key = f"unavailable:{widget_scope}"
        if unavailable_key in st.session_state:
            st.session_state[unavailable_key] = [
                item for item in st.session_state[unavailable_key] if item in model_names
            ]
        unavailable = settings_right.multiselect(
            "Phones currently unavailable",
            model_names,
            key=unavailable_key,
            help="Unavailable products remain in pipeline calculations so similar alternatives can be selected.",
        )

    current_key = build_request_key(
        signature, model_name, country, age, budget, total_units, unavailable
    )
    st.session_state["current_request_key"] = current_key
    saved = st.session_state.get("restock_result")
    matching_result = saved is not None and saved.get("key") == current_key

    generate = st.button("Generate recommendations", type="primary", key="generate_plan")
    if generate:
        st.session_state.pop("restock_result", None)
        try:
            with st.spinner("Training and generating recommendations..."):
                pipeline = train_pipeline(signature, model_name, data)
                plan = pipeline.recommend(
                    country,
                    int(age),
                    float(budget),
                    int(total_units),
                    unavailable,
                )
            st.session_state["restock_result"] = {
                "key": current_key,
                "plan": plan.copy(),
                "metrics": copy.deepcopy(pipeline.metrics),
                "mode": pipeline.mode,
                "context": {
                    "country": country,
                    "source_name": dataset_context["source_name"],
                    "source_note": dataset_context["source_note"],
                },
                "status": "empty" if plan.empty else "success",
            }
        except Exception as exc:
            st.error(f"Recommendations could not be generated: {exc}")
        saved = st.session_state.get("restock_result")
        matching_result = saved is not None and saved.get("key") == current_key

    st.divider()
    if not matching_result:
        st.header("2. Your results")
        if saved is None:
            st.info("Choose your preferences and generate recommendations.")
        else:
            st.info("Settings changed. Generate recommendations to update the results.")
        return

    if saved["status"] == "empty" or saved["plan"].empty:
        st.header("2. Your results")
        st.warning(
            "No profitable phones match these settings. Try a higher price limit or review the unavailable phones."
        )
        return
    render_results(saved, dataset_context)


def render_how_it_works_tab(dataset_context):
    st.header("How the tool works")
    st.caption("The application uses the existing Python pipeline for every recommendation.")
    st.subheader("1. Read the data")
    st.write(
        "The tool maps uploaded columns to a canonical schema, validates required product and price fields, "
        "keeps valid records, and reports rejected rows and missing-value policies."
    )
    st.subheader("2. Rank the phones")
    st.write(
        "When enough observed restock_success labels exist, XGBoost or Random Forest provides a supervised signal. "
        "Otherwise, a transparent score combines sentiment, ratings, and demand activity. Profit is included in the final ranking."
    )
    st.subheader("3. Forecast and split the stock")
    st.write(
        "The pipeline compares available one-step monthly forecasting methods with walk-forward validation. "
        "Forecast weights are converted into whole-unit quantities using largest-remainder allocation."
    )
    st.subheader("Costs and unavailable products")
    st.write(
        "Observed purchase costs are used when available; otherwise the existing pipeline applies documented fallbacks. "
        "Unavailable phones can be replaced with similar candidates using price, hardware ratings, ranking strength, and profit."
    )
    if dataset_context["ready"]:
        st.info(dataset_context["source_note"])
    else:
        st.info("Dataset-specific demand and model diagnostics are not available yet.")

    with st.expander("Model details and current diagnostics", expanded=False):
        saved = st.session_state.get("restock_result")
        current_key = st.session_state.get("current_request_key")
        if saved is None or not dataset_context["ready"]:
            st.write("Not evaluated yet. Generate recommendations to view current diagnostics.")
        elif current_key is None or saved["key"] != current_key:
            st.write("Settings or data changed. Generate recommendations again to refresh diagnostics.")
        else:
            render_diagnostics(saved["metrics"], saved["mode"], dataset_context["source_note"])


def apply_page_style():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1080px; padding-top: 2.25rem; padding-bottom: 2rem; }
        h1 { font-size: 2rem !important; letter-spacing: -0.02em; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1rem !important; }
        [data-testid="stCaptionContainer"] { color: #656c76; }
        a { color: #355580 !important; }
        hr { border-color: #dfe2e7 !important; }
        div[data-testid="stExpander"] { border-color: #dfe2e7; border-radius: 5px; }
        div[data-testid="stDataFrame"] { border: 1px solid #dfe2e7; border-radius: 5px; }
        @media (max-width: 720px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Mobile Restocking Tool", layout="wide")
    apply_page_style()
    title_col, link_col = st.columns([5, 1])
    title_col.title("Mobile Restocking Tool")
    title_col.caption("Find out which phones to stock and how many units to order.")
    link_col.markdown(f"[GitHub project]({REPOSITORY_URL})")

    recommendations_tab, dataset_tab, about_tab = st.tabs(
        ["Recommendations", "Dataset", "How it works"]
    )

    with dataset_tab:
        dataset_context = render_dataset_tab()
    with recommendations_tab:
        render_recommendations_tab(dataset_context)
    with about_tab:
        render_how_it_works_tab(dataset_context)


if __name__ == "__main__":
    main()
