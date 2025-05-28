import os
import pickle
import numpy as np
import streamlit as st

# 设置页面布局
st.set_page_config(
    page_title="AKI Probability Prediction",
    page_icon="🩺",
    layout="wide",
)

# 添加自定义 CSS 样式，增加页面边框
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

# 标题
st.title("🩺 AKI Probability Prediction")
st.markdown("Welcome to the Acute Kidney Injury Prediction Tool!")

# ========== 模型加载 ==========
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "lightgbm_model.pkl")

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"⚠️ 模型文件不存在：{path}")
        st.stop()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)

# ========== 输入特征设定 ==========
feature_specs = [
    ("Weight (kg)", 30.0, 300.0, 0),
    ("Length of Stay (days)", 1.0, 365.0, 0),
    ("SOFA Score", 0.0, 24.0, 0.0),
    ("Platelet Count (10^9/L)", 1.0, 1000.0, 0),
    ("Arterial BP Systolic (mmHg)", 50.0, 250.0, 0),
    ("SpO2 (%)", 50.0, 100.0, 50),
    ("Ventilator (0 = No, 1 = Yes)", 0, 1, 0),
]

# 主界面 - 输入框
st.header("🔧 Input Patient's Clinical Features")
st.write("Please fill in the following clinical measurements:")
cols = st.columns(len(feature_specs))
input_values = []

for idx, (name, min_v, max_v, default) in enumerate(feature_specs):
    label = name.split("(")[0].strip()

    # 判断是否为整数输入（排除布尔值）
    is_integer_input = all(isinstance(x, int) and not isinstance(x, bool) for x in [min_v, max_v, default])

    if is_integer_input:
        val = cols[idx].number_input(
            f"{label}",
            min_value=int(min_v),
            max_value=int(max_v),
            value=int(default),
            step=1,
            help=f"Enter {name}"
        )
    else:
        val = cols[idx].number_input(
            f"{label}",
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(default),
            step=(float(max_v) - float(min_v)) / 100.0,
            format="%.1f",
            help=f"Enter {name}"
        )
    input_values.append(val)

input_array = np.array(input_values).reshape(1, -1)

# 主界面 - 预测按钮
if st.button("🚀 Predict"):
    # 验证输入：除 SOFA Score 外，其他指标不能为 0
    invalid = any(input_values[i] == 0 for i in range(len(input_values)) if i != 2)
    if invalid:
        st.error("⚠️ Invalid input: Please ensure a valid value.")
    else:
        try:
            # 使用模型预测正类概率
            probability = model.predict(input_array)[0]

            # 显示预测结果
            st.subheader("🎯 Prediction Result")
            st.write(f"The predicted probability of AKI for this patient is: **{probability:.2%}**")

            # 解释提示
            if probability > 0.8:
                st.error("⚠️ High Risk: Immediate medical intervention is recommended!")
            elif probability > 0.5:
                st.warning("⚠️ Moderate Risk: Close monitoring is advised.")
            else:
                st.success("✅ Low Risk: No immediate action required.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
