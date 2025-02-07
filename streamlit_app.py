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
model_path = "lightgbm_model.pkl"
model = load_model(model_path)

# 定义输入特征（根据模型的特征）
feature_names = [
    "Weight (kg)",
    "Length of Stay (days)",
    "SOFA Score",
    "Platelet Count (10^9/L)",
    "Arterial BP Systolic (mmHg)",
    "SpO2 (%)",
    "Ventilator (0 = No, 1 = Yes)"
]

# 主界面 - 输入框
st.header("🔧 Input Patient's Clinical Features")
st.write("Please fill in the following clinical measurements:")

# 使用列布局来优化输入框的展示
columns = st.columns(len(feature_names))
input_values = []
for col, feature in zip(columns, feature_names):
    label = feature.split("(")[0].strip()  # 提取英文部分作为标签
    value = col.number_input(
        f"{label}",
        value=0.0,
        step=0.8,
        help=f"Enter {feature}"
    )
    input_values.append(value)

# 转换为 NumPy 数组
input_array = np.array(input_values).reshape(1, -1)

# 主界面 - 预测按钮
if st.button("🚀 Predict"):
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


# # 使用说明
# st.markdown("---")
# st.markdown("### 📖 使用说明：")
# st.markdown("""
# 1. 在页面顶部填写病人的医学特征值。
# 2. 确保输入值合理（例如，非负值）。
# 3. 点击 **预测** 按钮查看患病概率。
# 4. 根据预测结果采取相应的医学措施。
# """)
