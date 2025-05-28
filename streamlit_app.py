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
# 设置合理的最小/最大值，除 SOFA Score 和 Ventilator 可为 0 外，其它均需大于下限
feature_specs = [
    ("Weight (kg)",                  30.0, 200.0, 80.0),  # 成人常见体重范围
    ("Length of Stay (days)",         1.0, 365.0, 10.0), # 住院时长 1-365 天
    ("SOFA Score",                    0.0,  24.0,  2.0),  # SOFA 评分 0-24
    ("Platelet Count (10^9/L)",      50.0, 400.0, 300.0), # 血小板计数
    ("Arterial BP Systolic (mmHg)",  70.0, 200.0, 100.0), # 收缩压
    ("SpO2 (%)",                     70.0, 100.0,  95.0), # 血氧饱和度
    ("Ventilator (0 = No, 1 = Yes)",   0,     1,     0),   # 通气依赖
]

# 主界面 - 输入框
st.header("🔧 Input Patient's Clinical Features")
st.write("Please fill in the following clinical measurements:")
cols = st.columns(len(feature_specs))
input_values = []

for idx, (name, min_v, max_v, default) in enumerate(feature_specs):
    label = name.split("(")[0].strip()
    # 判断是否使用整数输入
    is_integer_input = isinstance(min_v, int) and isinstance(max_v, int) and isinstance(default, int)

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
    # 验证输入：除 SOFA（索引 2）和 Ventilator（索引 6）外，其它指标不能为 0
    invalid = any(
        val == 0 for i, val in enumerate(input_values) if i not in [2, 6]
    )
    if invalid:
        st.error("⚠️ 非法输入：除 SOFA 评分和通气依赖外，其他指标不能为 0，请重新填写。")
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
