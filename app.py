import os

import pandas as pd
import streamlit as st

from src.pipeline import InventoryPipeline
from src.schema import SchemaAdapter, SchemaError


st.set_page_config(page_title="Inventory Intelligence V2", layout="wide")
st.title("Inventory Intelligence V2")
st.caption("Demand forecasting, market scoring, sentiment-ready ingestion and stock allocation")

adapter = SchemaAdapter()
source = st.radio("Data source", ["Built-in dataset", "Upload dataset"], horizontal=True)

if source == "Built-in dataset":
    path = "Mobile Reviews Sentiment.csv"
    frame = pd.read_csv(path) if os.path.exists(path) else None
    if frame is None:
        st.error("The built-in dataset was not found.")
        st.stop()
    mapping = adapter.suggest_mapping(frame)
else:
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Upload a dataset to continue.")
        st.stop()
    frame = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    use_genai = st.checkbox("Use GenAI schema mapping", value=False)
    try:
        mapping = adapter.suggest_mapping(frame, use_genai=use_genai)
    except SchemaError as error:
        st.error(str(error))
        st.stop()

st.subheader("Schema mapping")
editable = pd.DataFrame([{"Source": column, "Mapped feature": mapping.get(column, "Ignore")} for column in frame.columns])
canonical = ["Ignore", "review_id", "brand", "model", "price_usd", "price_inr", "country", "age", "review_date", "review_text", "sentiment", "rating", "battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating", "units_sold", "purchase_cost"]
edited = st.data_editor(editable, hide_index=True, disabled=["Source"], column_config={"Mapped feature": st.column_config.SelectboxColumn(options=canonical)}, use_container_width=True)
mapping = {row["Source"]: row["Mapped feature"] for _, row in edited.iterrows() if row["Mapped feature"] != "Ignore"}
validation = adapter.validate(frame, mapping)

if validation.errors:
    for error in validation.errors:
        st.error(error)
    st.stop()
for warning in validation.warnings:
    st.warning(warning)
st.success(f"Dataset is compatible in {validation.mode} mode.")

try:
    data = adapter.transform(frame, mapping)
except SchemaError as error:
    st.error(str(error))
    st.stop()

countries = sorted(data["country"].dropna().astype(str).unique())
left, middle, right = st.columns(3)
country = left.selectbox("Country", countries)
age = middle.number_input("Target age", min_value=13, max_value=100, value=30)
budget = right.number_input("Maximum unit price", min_value=1.0, value=float(data["price_inr"].median()))
total_units = left.number_input("Total units", min_value=1, value=500)
model_name = middle.selectbox("Success model", ["xgboost", "random_forest"])
unavailable = right.multiselect("Unavailable models", sorted(data["model"].astype(str).unique()))

if st.button("Generate stocking plan", type="primary"):
    with st.spinner("Training and generating recommendations"):
        try:
            pipeline = InventoryPipeline(model_name=model_name).fit(data)
            plan = pipeline.recommend(country, age, budget, total_units, unavailable)
        except Exception as error:
            st.error(f"Pipeline failed: {error}")
            st.stop()
    if plan.empty:
        st.warning("No compatible products were found.")
    else:
        display = plan[["brand", "model", "price_inr", "success_probability", "score", "forecast", "suggested_quantity", "estimated_profit"]]
        st.dataframe(display, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Allocated units", int(plan["suggested_quantity"].sum()))
        c2.metric("Estimated revenue", f"INR {plan['estimated_revenue'].sum():,.0f}")
        c3.metric("Estimated profit", f"INR {plan['estimated_profit'].sum():,.0f}")
        st.bar_chart(plan.set_index("model")[["forecast"]])
        st.download_button("Download plan", plan.to_csv(index=False), "stocking_plan.csv", "text/csv")
