"""Feature engineering and the full modelling Pipeline for PriceSense."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Year the dataset was collected; used for car_age (single source of truth)
CURRENT_YEAR = 2025

# Columns the user (or the app) must provide
RAW_NUMERIC = ["car_age", "mileage_km", "engine_cc", "n_features", "is_registered"]
RAW_CATEGORICAL = ["make", "model", "fuel", "transmission", "assembly",
                   "body_type", "location"]
RAW_INPUTS = RAW_NUMERIC + RAW_CATEGORICAL

# Columns after feature engineering
NUMERIC = RAW_NUMERIC + ["mileage_per_year"]
CATEGORICAL = RAW_CATEGORICAL + ["brand_tier"]


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds mileage_per_year and brand_tier.

    fit() learns brand tiers from TRAINING rows only (median price per make),
    so no test information leaks in. Makes with fewer than `min_count` rows
    get the tier "Rare"; makes never seen in training get "Unlisted".
    """

    def __init__(self, min_count=50, cutoffs=(2.5e6, 5e6, 8e6, 20e6)):
        self.min_count = min_count
        self.cutoffs = cutoffs

    def fit(self, X, y=None):
        # y arrives on the log scale (TransformedTargetRegressor), so convert back
        price = pd.Series(np.expm1(np.asarray(y, dtype=float)), index=X.index)
        stats = price.groupby(X["make"]).agg(["count", "median"])
        c1, c2, c3, c4 = self.cutoffs
        tiers = {}
        for make, row in stats.iterrows():
            if row["count"] < self.min_count:
                tiers[make] = "Rare"
            elif row["median"] >= c4:
                tiers[make] = "Luxury"
            elif row["median"] >= c3:
                tiers[make] = "Premium"
            elif row["median"] >= c2:
                tiers[make] = "Upper-mid"
            elif row["median"] >= c1:
                tiers[make] = "Mid"
            else:
                tiers[make] = "Budget"
        self.tiers_ = tiers
        return self

    def transform(self, X):
        X = X.copy()
        X["mileage_per_year"] = X["mileage_km"] / X["car_age"].clip(lower=1)
        X["brand_tier"] = X["make"].map(self.tiers_).fillna("Unlisted")
        return X


def build_preprocessor(min_freq=20):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=min_freq)
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def build_pipeline(model):
    """features -> preprocessing -> model, as ONE sklearn Pipeline."""
    return Pipeline([("features", FeatureAdder()),
                     ("prep", build_preprocessor()),
                     ("model", model)])