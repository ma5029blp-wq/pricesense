"""PriceSense: used-car price predictor (Streamlit app)."""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
import features  # noqa: F401  (joblib needs it to unpickle add_features)

CURRENT_YEAR = 2025
st.set_page_config(page_title="PriceSense", page_icon="🚗", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#6C5CE7 0%,#00B4D8 100%);
       padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin:0; color:white; font-size:2.4rem;}
.hero p {margin:6px 0 0; color:rgba(255,255,255,.88); font-size:1.05rem;}
.card {background:#1A1F2B; border:1px solid #2B3245; border-radius:16px;
       padding:18px 20px; text-align:center;}
.card .label {color:#9AA4B8; font-size:.85rem; letter-spacing:.06em; text-transform:uppercase;}
.card .value {color:white; font-size:1.8rem; font-weight:700; margin-top:4px;}
.card.main {background:linear-gradient(135deg,#6C5CE7,#4834d4); border:none;}
.card.main .value {font-size:2.3rem;}
.badge {padding:12px 18px; border-radius:12px; font-weight:600; margin-top:14px;}
.good {background:#12372A; color:#4ADE80;}
.fair {background:#1E3A5F; color:#60A5FA;}
.over {background:#4A1D1D; color:#F87171;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "pricesense_pipeline.joblib")


@st.cache_data
def load_meta():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    opts = json.loads((ROOT / "models" / "options.json").read_text())
    imp = pd.read_csv(ROOT / "models" / "feature_importance.csv", index_col=0)["importance"]
    return meta, opts, imp


model = load_model()
meta, opts, importance = load_meta()
LOW, HIGH = meta["range_low_factor"], meta["range_high_factor"]
rf = next((r for r in meta["test_table"] if r.get("index") == "Random Forest"), {})

def default_index(options, value):
    """Position of `value` in `options` (0 if missing)."""
    return options.index(value) if value in options else 0


def fmt_pkr(x):
    """2,390,000 -> 'PKR 23.9 lacs' / 'PKR 1.27 crore'."""
    if x >= 10_000_000:
        return f"PKR {x / 10_000_000:.2f} crore"
    return f"PKR {x / 100_000:.1f} lacs"


def predict(df):
    return model.predict(df[features.RAW_INPUTS])


def card(label, value, main=False):
    cls = "card main" if main else "card"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div></div>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About PriceSense")
    st.write("Predicts a fair market price for used cars from PakWheels listings.")
    st.metric("Test R²", f"{rf.get('R2', 0.926):.3f}")
    st.metric("Median error", f"{rf.get('Median abs % error', 6.1):.1f}%")
    st.caption("Model: Random Forest in a single sklearn Pipeline. "
               "Metrics come from a held-out test set, evaluated once.")

st.markdown('<div class="hero"><h1>🚗 PriceSense</h1>'
            '<p>Know the fair price of a used car before you buy or sell.</p></div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔎 Single car", "📁 Batch (CSV)"])

# --------------------------------------------------------------- single car
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.subheader("Car")
        make = st.selectbox("Make", sorted(opts["models_by_make"]))
        model_name = st.selectbox("Model", opts["models_by_make"][make])
        year = st.number_input("Model year", opts["year_min"], opts["year_max"], 2018)
        mileage = st.number_input("Mileage (km)", 0, 500_000, 60_000, step=1000)
    with c2.container(border=True):
        st.subheader("Specs")
        default_cc = int(opts["engine_by_model"].get(model_name, 1300))
        engine = st.number_input("Engine (cc, 0 for electric)", 0, 7000, default_cc,
                                 step=50, key=f"engine_{model_name}")
        fuel = st.selectbox(
            "Fuel", opts["fuel"],
            index=default_index(opts["fuel"], opts["fuel_by_model"].get(model_name)),
            key=f"fuel_{model_name}")
        transmission = st.selectbox(
            "Transmission", opts["transmission"],
            index=default_index(opts["transmission"], opts["transmission_by_model"].get(model_name)),
            key=f"trans_{model_name}")
        assembly = st.selectbox(
            "Assembly", opts["assembly"],
            index=default_index(opts["assembly"], opts["assembly_by_model"].get(model_name)),
            key=f"assembly_{model_name}")
    with c3.container(border=True):
        st.subheader("Listing")
        body = st.selectbox(
            "Body type", opts["body_type"],
            index=default_index(opts["body_type"], opts["body_by_model"].get(model_name)),
            key=f"body_{model_name}")
        location = st.selectbox("Location", opts["location"])
        n_feat = st.slider("Number of listed features", 0, 28, opts["median_features"])
        listed = st.number_input("Listed price in PKR (optional)", 0, 500_000_000, 0,
                                 step=100_000, help="Enter an asking price to get a deal check.")

    if st.button("💰 Predict price", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "car_age": max(CURRENT_YEAR - year, 0), "mileage_km": mileage,
            "engine_cc": engine, "n_features": n_feat,
            "is_registered": int(location != "Unregistered"),
            "make": make, "model": model_name, "fuel": fuel,
            "transmission": transmission, "assembly": assembly,
            "body_type": body, "location": location,
        }])
        price = float(predict(row)[0])

        a, b, c = st.columns([1, 1.4, 1])
        a.markdown(card("Low estimate", fmt_pkr(price * LOW)), unsafe_allow_html=True)
        b.markdown(card("Predicted price", fmt_pkr(price), main=True), unsafe_allow_html=True)
        c.markdown(card("High estimate", fmt_pkr(price * HIGH)), unsafe_allow_html=True)
        st.caption("The range covers about 80% of typical prediction errors.")

        if listed > 0:
            ratio = listed / price
            if ratio < LOW:
                st.markdown(f'<div class="badge good">✅ Good deal: listed price is '
                            f'{ratio:.0%} of the predicted price.</div>', unsafe_allow_html=True)
            elif ratio > HIGH:
                st.markdown(f'<div class="badge over">⚠️ Overpriced: listed price is '
                            f'{ratio:.0%} of the predicted price.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge fair">ℹ️ Fair price: listed price is '
                            f'{ratio:.0%} of the predicted price.</div>', unsafe_allow_html=True)

    st.subheader("What drives the price?")
    imp_df = importance.head(10).reset_index()
    imp_df.columns = ["feature", "importance"]
    chart = (alt.Chart(imp_df)
             .mark_bar(cornerRadiusEnd=5, color="#6C5CE7")
             .encode(x=alt.X("importance:Q", title="Importance"),
                     y=alt.Y("feature:N", sort="-x", title=None),
                     tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")])
             .properties(height=330))
    st.altair_chart(chart, use_container_width=True)
    st.caption("Random Forest feature importance. 'n_features' also proxies for newer, higher-trim cars.")

# -------------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with these columns: " + ", ".join(features.RAW_INPUTS))
    sample = pd.DataFrame([{
        "car_age": 7, "mileage_km": 60000, "engine_cc": 1000, "n_features": 10,
        "is_registered": 1, "make": "Suzuki", "model": "Cultus", "fuel": "Petrol",
        "transmission": "Manual", "assembly": "Local", "body_type": "Hatchback",
        "location": "Lahore"}])
    st.download_button("⬇️ Download sample CSV", sample.to_csv(index=False), "sample.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in features.RAW_INPUTS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["predicted_price"] = predict(data).round(-3)
            data["range_low"] = (data["predicted_price"] * LOW).round(-3)
            data["range_high"] = (data["predicted_price"] * HIGH).round(-3)
            st.dataframe(data, use_container_width=True)
            st.download_button("⬇️ Download predictions", data.to_csv(index=False),
                               "predictions.csv")