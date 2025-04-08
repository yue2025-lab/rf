import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import io

# 忽略可能出现的警告
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="CVD Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        return joblib.load('rf.pkl')
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# 特征范围定义（修正了Hand_grip_strength的名称）
feature_ranges = {
    "Age": {"type": "numerical", "min": 0, "max": 100, "default": 62},
    "BMI": {"type": "numerical", "min": 10.00, "max": 50.00, "default": 23.24},
    "Waist": {"type": "numerical", "min": 50.00, "max": 140.00, "default": 92.30},
    "SBP": {"type": "numerical", "min": 50, "max": 220, "default": 147},
    "DBP": {"type": "numerical", "min": 40, "max": 180, "default": 90},
    "Triglyceride": {"type": "numerical", "min": 50.00, "max": 300.00, "default": 80.00},
    "RC": {"type": "numerical", "min": 0.00, "max": 5.00, "default": 0.25},
    "Platelet": {"type": "numerical", "min": 0, "max": 1000, "default": 140},
    "CRP": {"type": "numerical", "min": 0.00, "max": 100.00, "default": 0.45},
    "eGFR": {"type": "numerical", "min": 0.00, "max": 120.00, "default": 104.00},
    "Hand_grip_strength": {"type": "numerical", "min": 0.00, "max": 100.00, "default": 22.00},
}

# Streamlit 界面
st.title("心血管疾病风险预测模型")
st.markdown("### 请输入以下特征值进行预测")

# 动态生成输入项 - 使用两列布局
col1, col2 = st.columns(2)
feature_values = {}

for i, (feature, properties) in enumerate(feature_ranges.items()):
    # 交替在两列中显示输入项
    with col1 if i % 2 == 0 else col2:
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=f"input_{feature}"
            )
        elif properties["type"] == "categorical":
            value = st.selectbox(
                label=f"{feature} (Select a value)",
                options=properties["options"],
                key=f"input_{feature}"
            )
        feature_values[feature] = value

# 生成预测按钮
predict_button = st.button("预测风险", type="primary")

# 预测与 SHAP 可视化
if predict_button and model is not None:
    try:
        # 创建DataFrame而不是numpy数组，确保特征顺序正确
        input_df = pd.DataFrame([feature_values])
        
        # 模型预测
        predicted_class = model.predict(input_df)[0]
        predicted_proba = model.predict_proba(input_df)[0]
        
        # 提取预测的类别概率
        cvd_probability = predicted_proba[1] * 100  # 假设1表示有CVD风险
        
        # 设置风险级别
        if cvd_probability < 20:
            risk_level = "低风险"
            color = "green"
        elif cvd_probability < 50:
            risk_level = "中等风险"
            color = "orange"
        else:
            risk_level = "高风险"
            color = "red"
        
        # 使用Streamlit的原生功能显示结果
        st.markdown("## 预测结果")
        st.markdown(f"<h3 style='color:{color}'>心血管疾病风险概率: {cvd_probability:.2f}% ({risk_level})</h3>", unsafe_allow_html=True)
        
        # 计算SHAP值
        with st.spinner('计算特征影响...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # 为SHAP摘要图创建图形
            plt.figure(figsize=(10, 6))
            
            # 如果是二分类问题，选择类别1(有CVD风险)的SHAP值
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]  # 索引1表示有CVD的类别
            else:
                shap_values_to_plot = shap_values
                
            # 创建SHAP水平条形图
            shap.summary_plot(
                shap_values_to_plot, 
                input_df,
                plot_type="bar",
                show=False
            )
            plt.title("特征对预测的贡献度", fontsize=14)
            plt.tight_layout()
            
            # 将图形保存到内存中，然后显示
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_column_width=True)
            plt.close()
            
            # 创建SHAP力图的替代方案 - 使用决策图
            plt.figure(figsize=(12, 4))
            shap.decision_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                shap_values_to_plot,
                input_df,
                show=False
            )
            plt.title("特征如何影响预测结果", fontsize=14)
            plt.tight_layout()
            
            buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
            buf2.seek(0)
            st.image(buf2, use_column_width=True)
            plt.close()
            
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.exception(e)  # 显示详细的错误信息，方便调试

# 添加说明信息
with st.expander("模型说明"):
    st.markdown("""
    ### 关于模型
    - 此模型使用随机森林算法预测心血管疾病风险
    - 模型基于11个关键特征进行预测
    - AUC在外部验证集上为0.70
    
    ### 如何使用
    1. 输入患者的临床特征数据
    2. 点击"预测风险"按钮获取结果
    3. 查看风险概率和特征贡献度分析
    """)




