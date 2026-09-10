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

    def add_row(position, substitute_for="", similarity=1.0, substitute_score=None):
        row = ranked.iloc[position].copy()
        name = str(row["model"]).lower()
        if name in used:
            return False
        row["substitute_for"] = substitute_for
        row["substitution_similarity"] = float(similarity)
        row["substitute_score"] = float(row["score"] if substitute_score is None else substitute_score)
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
        distances, order = neighbors.kneighbors(values[position].reshape(1, -1))
        choices = []
        for distance, local_index in zip(distances[0], order[0]):
            replacement = eligible[int(local_index)]
            candidate = ranked.iloc[replacement]
            similarity = 1 / (1 + float(distance))
            business_score = (
                0.60 * similarity
                + 0.25 * float(candidate.get("success_probability", 0.5))
                + 0.15 * float(candidate.get("normalized_profit", 0.0))
            )
            choices.append((business_score, similarity, replacement))
        business_score, similarity, replacement = max(choices, key=lambda item: item[0])
        add_row(replacement, str(ranked.iloc[position]["model"]), similarity, business_score)

    if not selected:
        empty = ranked.iloc[0:0].copy()
        empty["substitute_for"] = pd.Series(dtype=str)
        empty["substitution_similarity"] = pd.Series(dtype=float)
        empty["substitute_score"] = pd.Series(dtype=float)
        return empty
    return pd.DataFrame(selected).sort_values(["substitute_score", "score"], ascending=False).head(limit).reset_index(drop=True)
