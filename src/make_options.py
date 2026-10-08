"""Build models/options.json: the dropdown options the Streamlit app needs.

Run from the project root:  python src/make_options.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "cleaned_cars.csv")

# models grouped by make (only models with 5+ listings, to keep the list clean)
mc = df.groupby(["make", "model"]).size()
models_by_make = {}
for (make, model), n in mc.items():
    if n >= 5:
        models_by_make.setdefault(make, []).append(model)
models_by_make = {m: sorted(v) for m, v in models_by_make.items()}

def most_common(col):
    """Most frequent value of `col` for each model."""
    return (df.groupby("model")[col]
            .agg(lambda s: s.value_counts().idxmax()).to_dict())

options = {
    "models_by_make": models_by_make,
    "fuel": sorted(df["fuel"].unique().tolist()),
    "transmission": sorted(df["transmission"].unique().tolist()),
    "assembly": sorted(df["assembly"].unique().tolist()),
    "body_type": sorted(df["body_type"].unique().tolist()),
    "location": sorted(df["location"].unique().tolist()),
    "engine_by_model": (df.groupby("model")["engine_cc"].median()
                        .dropna().round().astype(int).to_dict()),
    "body_by_model": most_common("body_type"),
    "fuel_by_model": most_common("fuel"),
    "transmission_by_model": most_common("transmission"),
    "assembly_by_model": most_common("assembly"),
    "year_min": int(df["year"].min()),
    "year_max": int(df["year"].max()),
    "median_features": int(df["n_features"].median()),
}
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "models" / "options.json").write_text(json.dumps(options))
print(len(models_by_make), "makes |",
      sum(len(v) for v in models_by_make.values()), "models")