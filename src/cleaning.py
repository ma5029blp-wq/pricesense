"""Cleaning functions for the PriceSense PakWheels dataset."""
import re
import ast
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025   # latest model year in the data; used for car_age
TWO_WORD_MAKES = ["Land Rover", "Range Rover", "Mercedes Benz", "Alfa Romeo",
                  "Rolls Royce", "Aston Martin", "Great Wall"]


def parse_price(text):
    """'PKR 1.27 crore' -> 12,700,000.  'Call for price' -> NaN."""
    m = re.search(r"([\d.,]+)\s*(lacs|crore)", str(text))
    if not m:
        return np.nan
    value = float(m.group(1).replace(",", ""))
    return value * (100_000 if m.group(2) == "lacs" else 10_000_000)


def to_number(series):
    """'120,000 km' -> 120000.  Anything unparseable -> NaN."""
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d]", "", regex=True),
                         errors="coerce")


def split_name(nam):
    """'Suzuki Alto VXL AGS 2022' -> ('Suzuki', 'Alto')."""
    nam = re.sub(r"\s*\b(19|20)\d{2}\s*$", "", str(nam)).strip()   # remove trailing year
    for make in TWO_WORD_MAKES:
        if nam.lower().startswith(make.lower() + " "):
            rest = nam[len(make):].split()
            return make, (rest[0] if rest else "Unknown")
    parts = nam.split()
    return parts[0], (parts[1] if len(parts) > 1 else "Unknown")


def count_features(text):
    """"['ABS', 'Air Bags']" (a string) -> 2.  Missing -> 0."""
    try:
        return len(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return 0


def clean_data(raw):
    df = raw.copy()

    # 1) duplicates and unusable rows
    df = df.drop_duplicates(subset="Ad Reference", keep="first")
    df["price"] = df["Price"].apply(parse_price)
    df = df.dropna(subset=["price"])            # 'Call for price'
    df = df[df["Year"] >= 1970]

    # 2) text -> numbers
    df["mileage_km"] = to_number(df["Millage"])
    df["engine_cc"] = to_number(df["Engine Capacity"])
    df["car_age"] = (CURRENT_YEAR - df["Year"]).clip(lower=0)

    # 3) impossible mileage -> NaN (imputed later inside the Pipeline)
    bad = (df["mileage_km"] > 500_000) | ((df["mileage_km"] < 1000) & (df["car_age"] > 3))
    df.loc[bad, "mileage_km"] = np.nan

    # 4) name -> make / model
    df[["make", "model"]] = df["nam"].apply(lambda x: pd.Series(split_name(x)))

    # 5) location, body type, features
    df["is_registered"] = (df["Province"] != "Un-Registered").astype(int)
    df["location"] = df["Province"].where(df["is_registered"] == 1, "Unregistered")
    df["body_type"] = df["Body Type"].fillna("Unknown")
    df["n_features"] = df["Features"].apply(count_features)

    # 6) keep only useful columns (IDs, url, owner name, color dropped)
    keep = ["price", "make", "model", "Year", "car_age", "mileage_km", "engine_cc",
            "Fuel", "Transmission", "Assembly", "body_type", "location",
            "is_registered", "n_features"]
    return df[keep].rename(columns={"Year": "year", "Fuel": "fuel",
                                    "Transmission": "transmission",
                                    "Assembly": "assembly"}).reset_index(drop=True)
