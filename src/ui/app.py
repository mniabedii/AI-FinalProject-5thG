import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np


# ---------------- Paths (repo structure) ----------------
ROOT = Path(__file__).resolve().parents[2]          # project root
MODELS_DIR = ROOT / "src" / "models"               # src/models

PREPROCESS_PATH = MODELS_DIR / "preprocess.pkl"

MODEL_DIRS = {
    "Model 1": MODELS_DIR / "model-1",   # contains: config.json + model-1.keras
    "Model 2": MODELS_DIR / "model-2",   # contains: config.json + model.keras
}

# raw features expected BEFORE one-hot/scaling (10 columns)
RAW_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

TARGET_COL = "Exited"   # change only if your dataset uses another name


# ---------------- Loaders (cached) ----------------
@st.cache_resource
def load_preprocess():
    return joblib.load(PREPROCESS_PATH)


@st.cache_resource
def load_model_and_config(model_dir: Path):
    cfg_path = model_dir / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    threshold = float(cfg.get("threshold", 0.5))

    # optional: if you saved features order in config
    # IMPORTANT: this must be the RAW feature order (10 cols), not after one-hot
    features = cfg.get("features", [])
    if not features:
        features = RAW_FEATURES

    candidate_models = ["model.keras", "model-1.keras"]
    model_path = None
    for name in candidate_models:
        p = model_dir / name
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(
            f"No model file found in {model_dir}. Expected one of: {candidate_models}"
        )

    model = tf.keras.models.load_model(model_path)
    return model, threshold, features


# ---------------- UI ----------------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Bank Customer Churn Prediction")
st.write("Fill customer details (one row) and predict churn probability and label.")

model_choice = st.selectbox("Choose model", list(MODEL_DIRS.keys()), index=0)

preprocess = load_preprocess()
model, threshold, features = load_model_and_config(MODEL_DIRS[model_choice])

st.caption(f"Using **{model_choice}** | Threshold = **{threshold:.2f}**")
st.caption(f"Preprocess: `{PREPROCESS_PATH}`")


# -------------- Inputs (raw 10 features) --------------
c1, c2 = st.columns(2)

with c1:
    CreditScore = st.number_input("CreditScore", min_value=0, max_value=1000, value=650, step=1)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=120, value=40, step=1)
    Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5, step=1)

with c2:
    Balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    NumOfProducts = st.number_input("NumOfProducts", min_value=1, max_value=10, value=2, step=1)
    HasCrCard = st.selectbox("HasCrCard", [0, 1], index=1)
    IsActiveMember = st.selectbox("IsActiveMember", [0, 1], index=1)
    EstimatedSalary = st.number_input("EstimatedSalary", min_value=0.0, value=80000.0, step=100.0, format="%.2f")

row = {
    "CreditScore": CreditScore,
    "Geography": Geography,
    "Gender": Gender,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary,
}

df = pd.DataFrame([row])

# enforce correct raw feature order
df = df[features]

st.divider()


# ---------------- Predict ----------------
if st.button("Predict", type="primary"):
    try:
        # preprocess expects raw 10 features
        X_trans = preprocess.transform(df)


        prob = float(model.predict(X_trans).ravel()[0])
        pred = int(prob >= threshold)

        st.subheader("Result")
        st.write(f"**Churn probability:** `{prob:.4f}`")
        st.write(f"**Prediction (threshold={threshold:.2f}):** `{pred}`")

        if pred == 1:
            st.error("High risk of churn ❗")
        else:
            st.success("Low risk of churn ✅")

        with st.expander("Debug details"):
            st.write("Input df columns:", df.columns.tolist())
            st.write("Transformed shape:", X_trans.shape)
            st.write("Transformed stats:",
                     {"min": float(np.min(X_trans)),
                      "max": float(np.max(X_trans)),
                      "mean": float(np.mean(X_trans))})
            st.dataframe(df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

