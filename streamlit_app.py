import streamlit as st
import pickle
import numpy as np

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
    /* 为页面主内容添加边框 */
    .stApp {
        padding: 1rem;   /* 调整内边距 */
        max-width: 90%;  /* 设置最大宽度 */
        margin: auto;    /* 居中内容 */
        border: 2px solid #4CAF50; /* 添加绿色边框 */
        border-radius: 10px; /* 设置边框圆角 */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* 添加阴影效果 */
        background-color: #f9f9f9; /* 添加背景色 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 设置标题
st.title("🩺 AKI Probability Prediction")
st.markdown("""
Welcome to the Acute Kidney Injury Prediction Tool!  
""")

# 加载训练好的模型
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# 模型路径
model_path = "D:/Python/project/CNN/mimic_proj/lightgbm_model.pkl"
model = load_model(model_path)

# 定义输入特征及其合理范围（根据临床经验设定）
feature_specs = [
    ("Weight (kg)", 2.0, 300.0, 70.0),
    ("Length of Stay (days)", 1.0, 365.0, 5.0),
    ("SOFA Score", 0.0, 24.0, 0.0),
    ("Platelet Count (10^9/L)", 1.0, 1000.0, 150.0),
    ("Arterial BP Systolic (mmHg)", 50.0, 250.0, 120.0),
    ("SpO2 (%)", 50.0, 100.0, 98.0),
    ("Ventilator (0 = No, 1 = Yes)", 0, 1, 0),
]

# 主界面 - 输入框
st.header("🔧 Input Patient's Clinical Features")
st.write("Please fill in the following clinical measurements:")

# 使用列布局来优化输入框的展示
cols = st.columns(len(feature_specs))
input_values = []
for idx, (name, min_v, max_v, default) in enumerate(feature_specs):
    label = name.split("(")[0].strip()
    # 区分整数和浮点数输入
    if isinstance(min_v, int) and isinstance(max_v, int):
        val = cols[idx].number_input(
            f"{label}",
            min_value=min_v,
            max_value=max_v,
            value=default,
            step=1,
            help=f"Enter {name}"
        )
    else:
        val = cols[idx].number_input(
            f"{label}",
            min_value=min_v,
            max_value=max_v,
            value=default,
            step=(max_v - min_v) / 100.0,
            format="%.1f",
            help=f"Enter {name}"
        )
    input_values.append(val)

# 转换为 NumPy 数组
input_array = np.array(input_values).reshape(1, -1)

# 主界面 - 预测按钮
if st.button("🚀 Predict"):
    # 验证输入：除 SOFA Score 外，其他指标不能为 0
    invalid = any(
        input_values[i] == 0 for i in range(len(input_values)) if i != 2
    )
    if invalid:
        st.error("⚠️ Invalid input: Please ensure a valid value.")
    else:
        try:
            # 使用模型预测正类的概率
            probability = model.predict(input_array)[0]  # LightGBM 返回正类概率

            # 显示预测结果
            st.subheader("🎯 Prediction Result")
            st.write(f"The predicted probability of AKI for this patient is: **{probability:.2%}**")

            # 结果解释
            if probability > 0.8:
                st.error("⚠️ High Risk: Immediate medical intervention is recommended!")
            elif probability > 0.5:
                st.warning("⚠️ Moderate Risk: Close monitoring is advised.")
            else:
                st.success("✅ Low Risk: No immediate action required.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
