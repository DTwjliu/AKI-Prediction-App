import os
import pickle
import numpy as np
import streamlit as st
import xgboost as xgb   # ✅ 加这一行
# Page configuration
st.set_page_config(
    page_title="AKI Probability Prediction",
    page_icon="🩺",
    layout="wide",
)

# Custom CSS for styling the app container
st.markdown(
    """
    <style>
    .stApp {
        padding: 1rem;
        max-width: 90%;
        margin: auto;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("🩺 AKI Probability Prediction")
st.markdown("Welcome to the Acute Kidney Injury Prediction Tool!")

# ========== Model Loading ==========
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

# ========== Input Feature Settings ==========
# New feature list requested by user:
# weight, mean urine output, SOFA score, Delta eGFR, diuretic use,
# mechanical ventilation, age, maximum BUN, APS III score, Delta BUN

# UI bounds (loose) and defaults are for convenience in the Streamlit widgets.
# clinical_ranges are used to validate values before prediction.
input_specs = [
    ("Weight (kg)", 0.0, 300.0, 70.0),
    ("Urine Output (mL/h)", 0.0, 2000.0, 50.0),
    ("SOFA", 0, 24, 0),
    ("Delta eGFR (mL/min/1.73m2)", -200.0, 200.0, 0.0),
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
    ("SOFA Score", 0, 24),
    ("Delta eGFR (mL/min/1.73m2)", -120.0, 120.0),
    ("Diuretic Use (0 = No, 1 = Yes)", 0, 1),
    ("Mechanical Ventilation (0 = No, 1 = Yes)", 0, 1),
    ("Age (years)", 18, 120),
    ("Maximum BUN (mg/dL)", 5.0, 150.0),
    ("APS III", 1, 200),
    ("Delta BUN (mg/dL)", -100.0, 100.0),
]

# Input section
st.header("🔧 Input Patient's Clinical Features")
st.write("Please enter the following clinical measurements:")
cols = st.columns(len(input_specs))
input_values = []

for idx, (name, min_ui, max_ui, default) in enumerate(input_specs):
    # Use the part before '(' for a shorter label in the UI
    label = name.split("(")[0].strip()
    # Detect integer-like UI bounds & defaults to present integer inputs
    is_integer = isinstance(min_ui, int) and isinstance(max_ui, int) and isinstance(default, int)

    if is_integer:
        value = cols[idx].number_input(
            label,
            min_value=int(min_ui),
            max_value=int(max_ui),
            value=int(default),
            step=1,
            help=f"Enter {name}"
        )
    else:
        # default step: 0.1 for floats to allow fine-grained clinical input
        step = (float(max_ui) - float(min_ui)) / 100.0
        if step <= 0:
            step = 0.1
        value = cols[idx].number_input(
            label,
            min_value=float(min_ui),
            max_value=float(max_ui),
            value=float(default),
            step=step,
            format="%.2f",
            help=f"Enter {name}"
        )
    input_values.append(value)

input_array = np.array(input_values).reshape(1, -1)

# Prediction button and validation
if st.button("🚀 Predict"):
    # Define indices that are allowed to be zero (SOFA, Delta eGFR, Diuretic, Mechanical Vent, Delta BUN)
    zero_allowed_indices = [2, 3, 4, 5, 9]

    # Zero-value validation (exclude indices in zero_allowed_indices)
    invalid_zero = any(
        (val == 0) for i, val in enumerate(input_values) if i not in zero_allowed_indices
    )

    # Range validation
    invalid_entries = []
    for i, val in enumerate(input_values):
        name, min_cl, max_cl = clinical_ranges[i]
        if val < min_cl or val > max_cl:
            invalid_entries.append(f"{name}: {val} (allowed {min_cl}-{max_cl})")

    if invalid_zero:
        st.error("⚠️ Invalid input: Except for SOFA Score, Delta eGFR, Diuretic Use, Mechanical Ventilation, and Delta BUN, all other values must be non-zero.")
    elif invalid_entries:
        st.error("⚠️ Input out of range: " + "; ".join(invalid_entries))
    else:
        try:
            prob = model.predict(input_array)[0]
            st.subheader("🎯 Prediction Result")
            st.write(f"Predicted AKI probability: **{prob:.2%}**")

            if prob > 0.8:
                st.error("High Risk: Immediate medical intervention recommended.")
            elif prob > 0.5:
                st.warning("Moderate Risk: Close monitoring advised.")
            else:
                st.success("Low Risk: No immediate action required.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
