import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SIMILARITY_FEATURES = ["price_inr", "rating", "battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]


def select_with_substitutes(scored, unavailable=None, limit=5):
    if scored.empty:
        return scored.copy()
    blocked = {str(value).lower() for value in (unavailable or [])}
    ranked = scored.sort_values("score", ascending=False).copy()
    feature_frame = ranked[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce")
    values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(feature_frame)
    values = StandardScaler().fit_transform(values)
    used = set()
    selected = []

    def add_row(position, substitute_for=""):
        row = ranked.iloc[position].copy()
        name = str(row["model"]).lower()
        if name in used:
            return False
        row["substitute_for"] = substitute_for
        selected.append(row)
        used.add(name)
        return True

    for position, row in enumerate(ranked.itertuples(index=False)):
        if len(selected) >= limit:
            break
        name = str(row.model).lower()
        if name in used:
            continue
        if name not in blocked:
            add_row(position)
            continue
        eligible = [
            index for index, candidate in enumerate(ranked["model"].astype(str).str.lower())
            if candidate not in blocked and candidate not in used
        ]
        if not eligible:
            continue
        neighbors = NearestNeighbors(n_neighbors=len(eligible), metric="euclidean")
        neighbors.fit(values[eligible])
        _, order = neighbors.kneighbors(values[position].reshape(1, -1))
        replacement = eligible[int(order[0][0])]
        add_row(replacement, str(ranked.iloc[position]["model"]))

    if not selected:
        return ranked.iloc[0:0].assign(substitute_for=pd.Series(dtype=str))
    return pd.DataFrame(selected).reset_index(drop=True)
