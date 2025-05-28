import os
import pickle
import numpy as np
import streamlit as st

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
MODEL_PATH = os.path.join(BASE_DIR, "lightgbm_model.pkl")

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
# Loose input limits for UI and strict clinical ranges for validation
input_specs = [
    ("Weight (kg)", 0.0, 300.0, 0.0),
    ("Length of Stay (days)", 0.0, 365.0, 0.0),
    ("SOFA Score", 0.0, 24.0, 0.0),
    ("Platelet Count (10^9/L)", 0.0, 1000.0, 0.0),
    ("Arterial BP Systolic (mmHg)", 0.0, 250.0, 0.0),
    ("SpO2 (%)", 0.0, 100.0, 0.0),
    ("Ventilator (0 = No, 1 = Yes)", 0, 1, 0),
]
clinical_ranges = [
    ("Weight (kg)", 30.0, 200.0),
    ("Length of Stay (days)", 1.0, 365.0),
    ("SOFA Score", 0.0, 24.0),
    ("Platelet Count (10^9/L)", 50.0, 400.0),
    ("Arterial BP Systolic (mmHg)", 70.0, 200.0),
    ("SpO2 (%)", 70.0, 100.0),
    ("Ventilator (0 = No, 1 = Yes)", 0, 1),
]

# Input section
st.header("🔧 Input Patient's Clinical Features")
st.write("Please enter the following clinical measurements:")
cols = st.columns(len(input_specs))
input_values = []

for idx, (name, min_ui, max_ui, default) in enumerate(input_specs):
    label = name.split("(")[0].strip()
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
        value = cols[idx].number_input(
            label,
            min_value=float(min_ui),
            max_value=float(max_ui),
            value=float(default),
            step=(float(max_ui) - float(min_ui)) / 100.0,
            format="%.1f",
            help=f"Enter {name}"
        )
    input_values.append(value)

input_array = np.array(input_values).reshape(1, -1)

# Prediction button and validation
if st.button("🚀 Predict"):
    # Zero-value validation (excluding SOFA and Ventilator)
    invalid_zero = any(
        val == 0 for i, val in enumerate(input_values) if i not in [2, 6]
    )
    # Range validation
    invalid_entries = []
    for i, val in enumerate(input_values):
        name, min_cl, max_cl = clinical_ranges[i]
        if val < min_cl or val > max_cl:
            invalid_entries.append(f"{name}: {val} (allowed {min_cl}-{max_cl})")

    if invalid_zero:
        st.error("⚠️ Invalid input: Except for SOFA Score and Ventilator, all other values must be non-zero.")
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
