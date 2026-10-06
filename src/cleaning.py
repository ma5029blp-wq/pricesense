"""Cleaning functions for the PriceSense PakWheels dataset."""
import re
import ast
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025
TWO_WORD_MAKES = ["Land Rover", "Range Rover", "Mercedes Benz", "Alfa Romeo",
                  "Rolls Royce", "Aston Martin", "Great Wall"]
# first word + second word that together form ONE model name
MODEL_PAIRS = {("land", "cruiser"), ("wagon", "r"), ("n", "box"), ("n", "wgn"),
               ("n", "one"), ("n", "van")}
JOIN_NEXT = {"class", "series"}      # "C Class", "7 Series"


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
    """'Honda N Box Custom GL 2022' -> ('Honda', 'N Box')."""
    nam = re.sub(r"\s*\b(19|20)\d{2}\s*$", "", str(nam)).strip()   # drop trailing year
    make, rest = None, []
    for m in TWO_WORD_MAKES:
        if nam.lower().startswith(m.lower() + " "):
            make, rest = m, nam[len(m):].split()
            break
    if make is None:
        parts = nam.split()
        make, rest = parts[0], parts[1:]
    if not rest:
        return make, "Unknown"
    model = rest[0]
    if len(rest) > 1:
        nxt = rest[1].lower()
        if nxt in JOIN_NEXT or (rest[0].lower(), nxt) in MODEL_PAIRS:
            model = f"{rest[0]} {rest[1]}"
    return make, model


def count_features(text):
    """"['ABS', 'Air Bags']" (a string) -> 2.  Missing -> 0."""
    try:
        return len(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return 0


def unify_case(series):
    """'Ek' and 'EK' -> whichever spelling is most common."""
    key = series.str.lower()
    canon = series.groupby(key).agg(lambda s: s.value_counts().idxmax())
    return key.map(canon)


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
    bad_km = (df["mileage_km"] > 500_000) | ((df["mileage_km"] < 1000) & (df["car_age"] > 3))
    df.loc[bad_km, "mileage_km"] = np.nan

    # 4) engine size: electric has no cc; typos in the rest -> NaN
    ev = df["Fuel"] == "Electric"
    df.loc[ev, "engine_cc"] = 0
    bad_cc = ~ev & ((df["engine_cc"] < 500) | (df["engine_cc"] > 7000))
    df.loc[bad_cc, "engine_cc"] = np.nan

    # 5) name -> make / model
    df[["make", "model"]] = df["nam"].apply(lambda x: pd.Series(split_name(x)))
    df["make"] = unify_case(df["make"])
    df["model"] = unify_case(df["model"])

    # 6) location, body type, features
    df["is_registered"] = (df["Province"] != "Un-Registered").astype(int)
    df["location"] = df["Province"].where(df["is_registered"] == 1, "Unregistered")
    df["body_type"] = df["Body Type"].fillna("Unknown")
    df["n_features"] = df["Features"].apply(count_features)

    keep = ["price", "make", "model", "Year", "car_age", "mileage_km", "engine_cc",
            "Fuel", "Transmission", "Assembly", "body_type", "location",
            "is_registered", "n_features"]
    return df[keep].rename(columns={"Year": "year", "Fuel": "fuel",
                                    "Transmission": "transmission",
                                    "Assembly": "assembly"}).reset_index(drop=True)
