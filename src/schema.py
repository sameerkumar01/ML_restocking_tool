import json
import os
from dataclasses import dataclass

import pandas as pd


class SchemaError(ValueError):
    pass


CANONICAL_TYPES = {
    "review_id": "string",
    "brand": "string",
    "model": "string",
    "price_usd": "number",
    "price_inr": "number",
    "country": "string",
    "age": "number",
    "review_date": "datetime",
    "review_text": "string",
    "sentiment": "string",
    "rating": "number",
    "battery_life_rating": "number",
    "camera_rating": "number",
    "performance_rating": "number",
    "design_rating": "number",
    "display_rating": "number",
    "units_sold": "number",
    "purchase_cost": "number",
}

ALIASES = {
    "product": "model",
    "product_name": "model",
    "mobile": "model",
    "mobile_name": "model",
    "manufacturer": "brand",
    "maker": "brand",
    "nation": "country",
    "market": "country",
    "customer_age": "age",
    "date": "review_date",
    "review_time": "review_date",
    "text": "review_text",
    "review": "review_text",
    "price": "price_inr",
    "cost": "price_inr",
    "sales": "units_sold",
    "quantity": "units_sold",
}


@dataclass
class ValidationResult:
    compatible: bool
    mode: str
    mapping: dict
    errors: list
    warnings: list


class SchemaAdapter:
    def profile(self, frame):
        return [{"name": str(column), "type": str(frame[column].dtype), "nullable": bool(frame[column].isna().any())} for column in frame.columns]

    def suggest_mapping(self, frame, use_genai=False, model="gemini-flash-latest"):
        mapping = self._deterministic_mapping(frame)
        if use_genai:
            mapping.update(self._genai_mapping(frame, model))
        return self._sanitize(mapping, frame.columns)

    def validate(self, frame, mapping):
        targets = set(mapping.values())
        errors = []
        warnings = []
        if "model" not in targets:
            errors.append("A product or model column is required.")
        if not ({"price_inr", "price_usd"} & targets):
            errors.append("A price column in INR or USD is required.")
        if "country" not in targets:
            warnings.append("Country is missing; values will default to Global.")
        has_time = "review_date" in targets
        has_demand = bool({"units_sold", "review_id"} & targets)
        has_text = "review_text" in targets
        has_sentiment = "sentiment" in targets
        if has_time and has_demand:
            mode = "full" if has_text or has_sentiment else "forecast"
        elif has_text or has_sentiment:
            mode = "review"
        else:
            mode = "recommendation"
        return ValidationResult(not errors, mode, mapping, errors, warnings)

    def transform(self, frame, mapping):
        result = frame.rename(columns=mapping).copy()
        if "country" not in result:
            result["country"] = "Global"
        for name, kind in CANONICAL_TYPES.items():
            if name not in result:
                continue
            if kind == "number":
                result[name] = pd.to_numeric(result[name], errors="coerce")
            elif kind == "datetime":
                result[name] = pd.to_datetime(result[name], errors="coerce")
            else:
                result[name] = result[name].astype("string")
        if "price_inr" not in result and "price_usd" in result:
            result["price_inr"] = result["price_usd"] * 87.0
        elif "price_usd" in result:
            result["price_inr"] = result["price_inr"].fillna(result["price_usd"] * 87.0)
        if "model" not in result or "price_inr" not in result:
            raise SchemaError("Missing required columns: model and price_inr are required.")
        invalid_model = result["model"].isna() | result["model"].str.strip().eq("")
        invalid_price = result["price_inr"].isna() | result["price_inr"].le(0)
        if invalid_model.any() or invalid_price.any():
            raise SchemaError("Required model and price values must be present and valid.")
        if "review_date" in result and ({"units_sold", "review_id"} & set(result.columns)) and result["review_date"].isna().any():
            raise SchemaError("Forecasting data contains missing or invalid review_date values.")
        return result

    def _deterministic_mapping(self, frame):
        mapping = {}
        for column in frame.columns:
            key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
            if key in CANONICAL_TYPES:
                mapping[column] = key
            elif key in ALIASES:
                mapping[column] = ALIASES[key]
        return mapping

    def _genai_mapping(self, frame, model):
        if not os.getenv("GOOGLE_API_KEY"):
            raise SchemaError("GOOGLE_API_KEY is required for Gemini schema mapping.")
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
        from pydantic import BaseModel, Field

        class MappingResponse(BaseModel):
            mapping: dict[str, str] = Field(default_factory=dict)

        payload = {"input": self.profile(frame), "allowed_output": CANONICAL_TYPES}
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Map source column names to allowed canonical features. Do not infer or transform row values. Return only the mapping."),
            ("human", "{payload}"),
        ])
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        chain = prompt | llm.with_structured_output(MappingResponse, method="json_schema")
        response = chain.invoke({"payload": json.dumps(payload)})
        if isinstance(response, dict):
            return response.get("mapping", {})
        return response.mapping

    def _sanitize(self, mapping, columns):
        source = {str(column): column for column in columns}
        clean = {}
        used = set()
        for raw, target in mapping.items():
            if str(raw) in source and target in CANONICAL_TYPES and target not in used:
                clean[source[str(raw)]] = target
                used.add(target)
        return clean
