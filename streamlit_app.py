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
# UI 特征顺序
# 按照 SHAP 特征重要性顺序展示
# =================================================
UI_FEATURE_NAMES = [
    "Weight",
    "Mean Urine Output",
    "SOFA",
    "Ventilation",
    "Delta eGFR",
    "Diuretics",
    "Age",
    "Max BUN",
    "Delta Bicarbonate",
    "Delta WBC",
]

# =================================================
# 模型训练时的原始特征顺序
# 传入模型前必须恢复为该顺序
# =================================================
DEFAULT_MODEL_FEATURE_NAMES = [
    "Delta WBC",
    "Mean Urine Output",
    "Delta eGFR",
    "Delta Bicarbonate",
    "Max BUN",
    "Ventilation",
    "Diuretics",
    "Age",
    "Weight",
    "SOFA",
]


def get_model_feature_names(loaded_model):
    """
    优先读取模型内部保存的特征名称和顺序。

    如果模型内部没有保存特征名，则使用训练时已知的默认顺序。
    """
    if hasattr(loaded_model, "feature_names"):
        feature_names = loaded_model.feature_names

        if feature_names is not None:
            return list(feature_names)

    if hasattr(loaded_model, "get_booster"):
        booster = loaded_model.get_booster()

        if booster.feature_names is not None:
            return list(booster.feature_names)

    return DEFAULT_MODEL_FEATURE_NAMES.copy()


MODEL_FEATURE_NAMES = get_model_feature_names(model)

# =================================================
# 检查 UI 特征与模型特征是否完全一致
# 仅允许顺序不同，不允许名称缺失或额外增加
# =================================================
missing_in_ui = [
    feature
    for feature in MODEL_FEATURE_NAMES
    if feature not in UI_FEATURE_NAMES
]

extra_in_ui = [
    feature
    for feature in UI_FEATURE_NAMES
    if feature not in MODEL_FEATURE_NAMES
]

if missing_in_ui or extra_in_ui:
    error_messages = []

    if missing_in_ui:
        error_messages.append(
            "Missing UI features required by the model: "
            + ", ".join(missing_in_ui)
        )

    if extra_in_ui:
        error_messages.append(
            "Unexpected UI features not used by the model: "
            + ", ".join(extra_in_ui)
        )

    st.error("⚠️ Feature configuration error: " + " | ".join(error_messages))
    st.stop()

# =================================================
# Input settings
# UI 顺序按照 SHAP 特征重要性排序
# =================================================
input_specs = [
    ("Weight (kg)", 0.0, 300.0, 70.0),
    ("Mean Urine Output (mL/h)", 0.0, 2000.0, 50.0),
    ("SOFA", 0, 24, 0),
    ("Ventilation (0 = No, 1 = Yes)", 0, 1, 0),
    ("Delta eGFR (mL/min/1.73m²)", -200.0, 200.0, 0.0),
    ("Diuretic Use (0 = No, 1 = Yes)", 0, 1, 0),
    ("Age (years)", 0, 120, 65),
    ("Max BUN (mg/dL)", 0.0, 300.0, 20.0),
    ("Delta Bicarbonate (mmol/L)", -50.0, 50.0, 0.0),
    ("Delta WBC (10^9/L)", -20.0, 20.0, 0.0),
]

clinical_ranges = [
    ("Weight (kg)", 30.0, 200.0),
    ("Mean Urine Output (mL/h)", 10.0, 2000.0),
    ("SOFA", 0, 24),
    ("Ventilation", 0, 1),
    ("Delta eGFR", -120.0, 120.0),
    ("Diuretic Use", 0, 1),
    ("Age (years)", 18, 120),
    ("Max BUN (mg/dL)", 5.0, 150.0),
    ("Delta Bicarbonate", -30.0, 30.0),
    ("Delta WBC", -15.0, 15.0),
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
            step=1,
        )
    else:
        value = cols[idx].number_input(
            label,
            min_value=float(min_ui),
            max_value=float(max_ui),
            value=float(default),
            step=0.1,
            format="%.2f",
        )

    input_values.append(value)

# =================================================
# Prediction
# =================================================
if st.button("🚀 Predict"):
    # UI 顺序中允许为 0 的特征：
    # SOFA、Ventilation、Delta eGFR、Diuretics、
    # Delta Bicarbonate、Delta WBC
    zero_allowed_indices = [2, 3, 4, 5, 8, 9]

    invalid_zero = any(
        val == 0
        for i, val in enumerate(input_values)
        if i not in zero_allowed_indices
    )

    invalid_entries = []

    for i, val in enumerate(input_values):
        name, min_cl, max_cl = clinical_ranges[i]

        if val < min_cl or val > max_cl:
            invalid_entries.append(
                f"{name}: {val} (allowed {min_cl}-{max_cl})"
            )

    if invalid_zero:
        st.error("⚠️ Invalid input: some values cannot be zero.")

    elif invalid_entries:
        st.error(
            "⚠️ Input out of range: "
            + "; ".join(invalid_entries)
        )

    else:
        try:
            # =================================================
            # 第一步：按照 UI 顺序，将用户输入映射到特征名称
            # =================================================
            input_data = {
                "Weight": input_values[0],
                "Mean Urine Output": input_values[1],
                "SOFA": input_values[2],
                "Ventilation": input_values[3],
                "Delta eGFR": input_values[4],
                "Diuretics": input_values[5],
                "Age": input_values[6],
                "Max BUN": input_values[7],
                "Delta Bicarbonate": input_values[8],
                "Delta WBC": input_values[9],
            }

            # =================================================
            # 第二步：先构建 DataFrame
            # 此时列顺序仍然是 UI 的 SHAP 顺序
            # =================================================
            input_df = pd.DataFrame([input_data])

            # =================================================
            # 第三步：预测前重新排列为模型训练时的原始顺序
            # =================================================
            input_df = input_df.reindex(
                columns=MODEL_FEATURE_NAMES
            )

            # 检查重新排列后是否出现缺失列或缺失值
            if input_df.columns.tolist() != MODEL_FEATURE_NAMES:
                raise ValueError(
                    "The input feature order could not be aligned "
                    "with the model feature order."
                )

            if input_df.isnull().any().any():
                missing_features = input_df.columns[
                    input_df.isnull().any()
                ].tolist()

                raise ValueError(
                    "Missing values detected for model features: "
                    + ", ".join(missing_features)
                )

            # =================================================
            # 第四步：建立 DMatrix
            # 此时列名和顺序与训练模型完全一致
            # =================================================
            d_input = xgb.DMatrix(
                input_df,
                feature_names=MODEL_FEATURE_NAMES,
            )

            # =================================================
            # 第五步：模型预测
            # =================================================
            prob = float(model.predict(d_input)[0])

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
                st.error(
                    "High Risk: Immediate medical intervention recommended."
                )
            elif prob >= 0.5:
                st.warning(
                    "Moderate Risk: Close monitoring advised."
                )
            else:
                st.success(
                    "Low Risk: No immediate action required."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
