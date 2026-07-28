"""
Interactive Gallstone Risk Prediction — Streamlit App

Enter patient biometric/lab values, get a live risk score from the
tuned Random Forest model, plus a SHAP waterfall explaining exactly
why the model scored that patient the way it did.

Run: streamlit run app.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Gallstone Risk Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model_and_info():
    model = joblib.load("app_model.joblib")
    with open("app_feature_info.json") as f:
        info = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, info, explainer


model, info, explainer = load_model_and_info()
feature_order = info["feature_order"]
medians = info["medians"]

BINARY_FIELDS = {
    "gender": ("Gender", {0: "Female", 1: "Male"}),
    "cad": ("Coronary Artery Disease", {0: "No", 1: "Yes"}),
    "hypothyroidism": ("Hypothyroidism", {0: "No", 1: "Yes"}),
    "hyperlipidemia": ("Hyperlipidemia", {0: "No", 1: "Yes"}),
    "diabetes": ("Diabetes Mellitus", {0: "No", 1: "Yes"}),
}

st.title("🩺 Gallstone Risk Predictor")
st.caption(
    "Portfolio project — trained on the UCI Gallstone Disease dataset (n=319). "
    "This is a demonstration of an ML pipeline, **not medical advice**."
)

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
This app is the interactive layer on top of a SQL + Python analytics
project:

1. Raw clinical data → SQLite
2. SQL risk analysis (window functions, CTEs, correlated subqueries)
3. Model comparison + hyperparameter tuning (Random Forest, ROC-AUC ≈ 0.89)
4. SHAP explainability — shown live below for whatever values you enter

**Key finding:** CRP (an inflammation marker) and Vitamin D are stronger
predictors here than BMI or cholesterol.

[View the full project on GitHub](https://github.com/alexh30486-ui/Predicting-Gallstone-Disease-from-Body-Composition-Blood-Chemistry)
"""
    )

st.subheader("Patient inputs")
st.caption("Adjust the values below — these are the features that matter most to the model.")

col1, col2, col3 = st.columns(3)
values = dict(medians)  # start every feature at the dataset median

with col1:
    values["age"] = st.slider("Age", 18, 96, int(medians["age"]))
    gender_label = st.selectbox("Gender", ["Female", "Male"])
    values["gender"] = 1 if gender_label == "Male" else 0
    values["bmi"] = st.slider("BMI", 15.0, 45.0, float(round(medians["bmi"], 1)))
    values["crp"] = st.slider("C-Reactive Protein (CRP, mg/L)", 0.0, 50.0, float(round(medians["crp"], 1)))

with col2:
    values["vitamin_d"] = st.slider("Vitamin D (ng/mL)", 3.0, 90.0, float(round(medians["vitamin_d"], 1)))
    values["ast"] = st.slider("AST (U/L)", 5.0, 100.0, float(round(medians["ast"], 1)))
    values["total_cholesterol"] = st.slider(
        "Total Cholesterol (mg/dL)", 100.0, 350.0, float(round(medians["total_cholesterol"], 1))
    )
    diabetes_label = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
    values["diabetes"] = 1 if diabetes_label == "Yes" else 0

with col3:
    values["visceral_fat_area"] = st.slider(
        "Visceral Fat Area", 20.0, 250.0, float(round(medians["visceral_fat_area"], 1))
    )
    values["comorbidity"] = st.selectbox("Number of Comorbidities", [0, 1, 2, 3], index=int(medians["comorbidity"]))
    hyperlipidemia_label = st.selectbox("Hyperlipidemia", ["No", "Yes"])
    values["hyperlipidemia"] = 1 if hyperlipidemia_label == "Yes" else 0
    cad_label = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
    values["cad"] = 1 if cad_label == "Yes" else 0

with st.expander("Advanced: full biometric & lab panel (defaults to dataset median)"):
    adv_cols = st.columns(4)
    remaining = [f for f in feature_order if f not in values or f in
                 ("comorbidity", "cad", "hyperlipidemia", "diabetes")]
    # (the four above are already set from the main form; skip re-rendering them)
    remaining = [f for f in feature_order if f not in
                 ("age", "gender", "bmi", "crp", "vitamin_d", "ast", "total_cholesterol",
                  "diabetes", "visceral_fat_area", "comorbidity", "hyperlipidemia", "cad")]
    for i, feat in enumerate(remaining):
        col = adv_cols[i % 4]
        with col:
            if feat in BINARY_FIELDS:
                label, mapping = BINARY_FIELDS[feat]
                sel = st.selectbox(label, list(mapping.values()), key=f"adv_{feat}")
                values[feat] = [k for k, v in mapping.items() if v == sel][0]
            else:
                lo, hi = info["mins"][feat], info["maxs"][feat]
                default = medians[feat]
                values[feat] = st.number_input(
                    feat.replace("_", " ").title(), value=float(round(default, 2)),
                    min_value=float(lo), max_value=float(hi), key=f"adv_{feat}"
                )

st.divider()

if st.button("Predict Risk", type="primary"):
    X_input = pd.DataFrame([[values[f] for f in feature_order]], columns=feature_order)
    proba = model.predict_proba(X_input)[0, 1]

    result_col, chart_col = st.columns([1, 2])

    with result_col:
        st.metric("Predicted Gallstone Risk", f"{proba * 100:.1f}%")
        if proba >= 0.6:
            st.error("Model flags this as **higher risk**.")
        elif proba >= 0.4:
            st.warning("Model flags this as **borderline / uncertain**.")
        else:
            st.success("Model flags this as **lower risk**.")

    with chart_col:
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_value = explainer.expected_value[1]
        elif np.ndim(shap_values) == 3:
            sv = shap_values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value

        explanation = shap.Explanation(
            values=sv,
            base_values=base_value,
            data=X_input.iloc[0].values,
            feature_names=feature_order,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption(
        "The waterfall shows how each value pushed the prediction above or below "
        "the model's baseline (average) prediction — this is the same SHAP "
        "technique used in the full project notebook."
    )
else:
    st.info("Set the values above and click **Predict Risk** to see a live SHAP explanation.")
