import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
import features

# 1) split BEFORE any fitting
df = pd.read_csv(ROOT / "data" / "cleaned_cars.csv")
X, y = df[features.RAW_INPUTS], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 2) compare models with cross-validation (train set only)
models = {
    "Baseline (median)": DummyRegressor(strategy="median"),
    "Linear (Ridge)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                           n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                   random_state=42),
}
cv = KFold(n_splits=3, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    t = time.time()
    res = cross_validate(features.build_pipeline(clone(m)), X_train, y_train_log,
                         cv=cv, n_jobs=1,
                         scoring=["neg_mean_absolute_error",
                                  "neg_root_mean_squared_error", "r2"],
                         params=None)
    rows.append({"model": name,
                 "MAE (log)": -res["test_neg_mean_absolute_error"].mean(),
                 "RMSE (log)": -res["test_neg_root_mean_squared_error"].mean(),
                 "R2": res["test_r2"].mean(),
                 "seconds": time.time() - t})
    print(name, "done")
cv_table = pd.DataFrame(rows).round(3).set_index("model")
print("\nCross-validation (log scale, train only):\n", cv_table)

best_name = cv_table.drop("Baseline (median)")["RMSE (log)"].idxmin()
print("\nBest model:", best_name)

# 3) price range from out-of-fold residuals
cv_pred = cross_val_predict(features.build_pipeline(clone(models[best_name])),
                            X_train, y_train_log, cv=cv, n_jobs=1)
q10, q90 = np.quantile(y_train_log - cv_pred, [0.10, 0.90])


def make_final(m):
    return TransformedTargetRegressor(regressor=features.build_pipeline(clone(m)),
                                      func=np.log1p, inverse_func=np.expm1)


# 4) final fit, then the ONE test evaluation
final = make_final(models[best_name]).fit(X_train, y_train)
baseline = make_final(models["Baseline (median)"]).fit(X_train, y_train)


def report(m):
    p = m.predict(X_test)
    return {"MAE (PKR)": mean_absolute_error(y_test, p),
            "RMSE (PKR)": mean_squared_error(y_test, p) ** 0.5,
            "R2": r2_score(y_test, p),
            "Median abs % error": float(np.median(np.abs(p - y_test) / y_test) * 100)}


test_table = pd.DataFrame({"Baseline (median)": report(baseline),
                           best_name: report(final)}).T.round(3)
print("\nFINAL TEST RESULTS:\n", test_table)

# 5) save model, feature importance, metadata
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(final, ROOT / "models" / "pricesense_pipeline.joblib", compress=3)

inner = final.regressor_
est = inner.named_steps["model"]
names = inner.named_steps["prep"].get_feature_names_out()


def original_column(enc):
    base = enc.split("__", 1)[1]
    for col in sorted(features.NUMERIC + features.CATEGORICAL, key=len, reverse=True):
        if base == col or base.startswith(col + "_"):
            return col
    return base


importance = (pd.Series(est.feature_importances_, index=names)
              .groupby(original_column).sum().sort_values(ascending=False))
importance.to_csv(ROOT / "models" / "feature_importance.csv", header=["importance"])

meta = {"best_model": best_name,
        "range_low_factor": float(np.exp(q10)), "range_high_factor": float(np.exp(q90)),
        "cv_table": cv_table.reset_index().to_dict("records"),
        "test_table": test_table.reset_index().to_dict("records")}
(ROOT / "models" / "meta.json").write_text(json.dumps(meta, indent=2))
print("\nTop features:\n", importance.head(10).round(3))
print("Model size (MB):", round((ROOT / "models" / "pricesense_pipeline.joblib").stat().st_size / 1e6, 1))