"""Feature engineering and the full modelling Pipeline for PriceSense."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Brand tiers come from median price per make in the EDA (price levels, not quality)
BRAND_TIERS = {
    "Luxury":    ["Range Rover", "Porsche", "Ford", "Lexus"],
    "Premium":   ["Audi", "Haval", "JAC", "BYD", "BMW", "Mercedes Benz", "BAIC"],
    "Upper-mid": ["Chery", "MG", "KIA", "Peugeot", "Hyundai", "DFSK"],
    "Mid":       ["Toyota", "Proton", "Honda", "Changan", "Nissan", "Mazda"],
    "Budget":    ["Daihatsu", "Subaru", "Mitsubishi", "Jeep", "Suzuki", "FAW",
                  "Prince", "Chevrolet", "United", "Daewoo"],
}
MAKE_TO_TIER = {m: tier for tier, makes in BRAND_TIERS.items() for m in makes}

# Columns the user (or the app) must provide
RAW_NUMERIC = ["car_age", "mileage_km", "engine_cc", "n_features", "is_registered"]
RAW_CATEGORICAL = ["make", "model", "fuel", "transmission", "assembly",
                   "body_type", "location"]
RAW_INPUTS = RAW_NUMERIC + RAW_CATEGORICAL

# Columns after add_features
NUMERIC = RAW_NUMERIC + ["mileage_per_year"]
CATEGORICAL = RAW_CATEGORICAL + ["brand_tier"]


def add_features(df):
    """Row-by-row features. Nothing is learned from data, so no leakage."""
    df = df.copy()
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].clip(lower=1)
    df["brand_tier"] = df["make"].map(MAKE_TO_TIER).fillna("Unlisted")
    return df


def build_preprocessor(min_freq=20):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=min_freq)
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def build_pipeline(model):
    """features -> preprocessing -> model, as ONE sklearn Pipeline."""
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", build_preprocessor()),
                     ("model", model)])