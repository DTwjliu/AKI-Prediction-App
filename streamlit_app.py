import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb

# =================================================
# Page configuration
# =================================================
st.set_page_config(
    page_title="AKI Probability Prediction",
    page_icon="🩺",
    layout="wide",
)

# =================================================
# Custom CSS
# =================================================
st.markdown(
    """
    <style>
    /* 全局背景 */
    .stApp {
        padding: 1.5rem;
        max-width: 92%;
        margin: auto;
        background: linear-gradient(135deg, #f8fbff 0%, #f4f9f4 100%);
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    /* 标题 */
    h1 {
        color: #1f4e79;
        font-weight: 700;
        margin-bottom: 0.3em;
    }

    /* 二级标题 */
    h2, h3 {
        color: #2c6e49;
        font-weight: 600;
    }

    /* 输入区域卡片 */
    section[data-testid="stVerticalBlock"] > div:has(> div.stColumns) {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.5rem 1.5rem 1.5rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }

    /* 输入框 */
    input {
        border-radius: 8px !important;
        border: 1px solid #cfd8dc !important;
        padding: 6px 10px !important;
        transition: all 0.2s ease-in-out;
    }

    input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.25) !important;
    }

    /* 按钮 */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4CAF50, #43a047);
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        border: none;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.35);
        transition: all 0.2s ease-in-out;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(76, 175, 80, 0.45);
    }

    /* 结果卡片 */
    .result-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10);
        border-left: 6px solid #4CAF50;
    }

    /* 风险标签 */
    .risk-high {
        color: #b71c1c;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .risk-medium {
        color: #ef6c00;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .risk-low {
        color: #2e7d32;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================
# Title
# =================================================
st.title("🩺 AKI Probability Prediction")
st.markdown("Welcome to the Acute Kidney Injury Prediction Tool!")

# =================================================
# Model Loading
# =================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"⚠️ Model file not found: {path}")
        st.stop()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)

# =================================================
# ⚠️ 特征名（必须与训练时完全一致）
# =================================================
FEATURE_NAMES = [
    "Mean Urine Output",
    "Delta eGFR",
    "Max BUN",
    "Delta BUN",
    "Ventilation",
    "Diuretics",
    "Age",
    "Weight",
    "APS III",
    "SOFA"
]

# =================================================
# Input settings (UI 顺序可以随意)
# =================================================
input_specs = [
    ("Weight (kg)", 0.0, 300.0, 70.0),
    ("Urine Output (mL/h)", 0.0, 2000.0, 50.0),
    ("SOFA", 0, 24, 0),
    ("Delta eGFR (mL/min/1.73m²)", -200.0, 200.0, 0.0),
    ("Diuretic Use (0 = No, 1 = Yes)", 0, 1, 0),
    ("Mechanical Ventilation (0 = No, 1 = Yes)", 0, 1, 0),
    ("Age (years)", 0, 120, 65),
    ("Maximum BUN (mg/dL)", 0.0, 300.0, 20.0),
    ("APS III", 0, 300, 50),
    ("Delta BUN (mg/dL)", -200.0, 200.0, 0.0),
]

clinical_ranges = [
    ("Weight (kg)", 30.0, 200.0),
    ("Urine Output (mL/h)", 10.0, 2000.0),
    ("SOFA", 0, 24),
    ("Delta eGFR", -120.0, 120.0),
    ("Diuretic Use", 0, 1),
    ("Mechanical Ventilation", 0, 1),
    ("Age (years)", 18, 120),
    ("Maximum BUN (mg/dL)", 5.0, 150.0),
    ("APS III", 1, 200),
    ("Delta BUN", -100.0, 100.0),
]

# =================================================
# Input UI
# =================================================
st.header("🔧 Input Patient Clinical Features")
cols = st.columns(len(input_specs))
input_values = []

for idx, (name, min_ui, max_ui, default) in enumerate(input_specs):
    label = name.split("(")[0].strip()
    is_integer = isinstance(min_ui, int) and isinstance(max_ui, int)

    if is_integer:
        value = cols[idx].number_input(
            label,
            min_value=int(min_ui),
            max_value=int(max_ui),
            value=int(default),
            step=1
        )
    else:
        value = cols[idx].number_input(
            label,
            min_value=float(min_ui),
            max_value=float(max_ui),
            value=float(default),
            step=0.1,
            format="%.2f"
        )
    input_values.append(value)

# =================================================
# Prediction
# =================================================
if st.button("🚀 Predict"):
    zero_allowed_indices = [2, 3, 4, 5, 9]

    invalid_zero = any(
        (val == 0) for i, val in enumerate(input_values)
        if i not in zero_allowed_indices
    )

    invalid_entries = []
    for i, val in enumerate(input_values):
        name, min_cl, max_cl = clinical_ranges[i]
        if val < min_cl or val > max_cl:
            invalid_entries.append(f"{name}: {val} (allowed {min_cl}-{max_cl})")

    if invalid_zero:
        st.error("⚠️ Invalid input: some values cannot be zero.")
    elif invalid_entries:
        st.error("⚠️ Input out of range: " + "; ".join(invalid_entries))
    else:
        try:
            # -------------------------------------------------
            # 🚨 核心修复：用 DataFrame + 正确特征名
            # -------------------------------------------------
            input_df = pd.DataFrame(
                [[
                    input_values[1],  # Mean Urine Output
                    input_values[3],  # Delta eGFR
                    input_values[7],  # Max BUN
                    input_values[9],  # Delta BUN
                    input_values[5],  # Ventilation
                    input_values[4],  # Diuretics
                    input_values[6],  # Age
                    input_values[0],  # Weight
                    input_values[8],  # APS III
                    input_values[2],  # SOFA
                ]],
                columns=FEATURE_NAMES
            )

            d_input = xgb.DMatrix(input_df)
            prob = model.predict(d_input)[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🎯 Prediction Result</h3>
                    <p style="font-size: 1.3rem;">
                    Predicted AKI Probability:
                    <strong>{prob:.2%}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


            if prob >= 0.8:
                st.error("High Risk: Immediate medical intervention recommended.")
            elif prob >= 0.5:
                st.warning("Moderate Risk: Close monitoring advised.")
            else:
                st.success("Low Risk: No immediate action required.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
