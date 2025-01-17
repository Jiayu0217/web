import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# 加载模型
@st.cache_resource
def load_model():
    model = joblib.load('D:\\desktop\\333\\best_xgb_model.pkl')
    return model


model = load_model()

# 创建Streamlit应用
st.title('***预测')

# 创建输入字段
st.header('请输入信息：')
feature_names = ['Heart valve surgery', 'Sex', 'Hypotension',
       'Electrolyte and acid-base imbalance',
       'Non-rheumatic valvular heart disease', 'Liver failure',
       'Respiratory failure', 'Acute renal failure',
       'Chronic rheumatic heart disease',
       'Chronic obstructive pulmonary disease (COPD)', 'Cerebral hemorrhage',
       'Cerebral infarction', 'Glomerular disease', 'Infective endocarditis',
       'Heart failure', 'Cardiac arrest', 'Emphysema or chronic bronchitis',
       'Asthma', 'Antiplatelet drugs', 'NSAIDs', 'PTCA', 'Aortic surgery',
       'CRP', 'Age (at the time of enrollment)', 'BMI', 'Cystatin C',
       'Blood glucose', 'Red blood cell count', 'Total cholesterol',
       'Triglycerides', 'Albumin', 'IGF-1', 'Gamma-glutamyl transferase (GGT)',
       'NLR', 'Smoking status_0', 'Smoking status_2',
       'Alcohol consumption status_4']  # 替换为你的特征名称
values=[ 0.        ,  1.        ,  0.        ,  1.        ,  1.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  2.83      , 63.        , 27.1977    ,
        1.095     ,  4.647     ,  4.44      ,  6.654     ,  1.943     ,
       43.01      , 21.424     , 52.6       ,  2.17647059,  0.        ,
        1.        ,  0.        ]
features = {}

for i,feature in enumerate(feature_names):
    features[feature] = st.number_input(f'{feature}:', value=values[i])
scale=joblib.load('D:\\desktop\\333\\scaler.pkl')

# 创建预测按钮
if st.button('预测'):
    # 将输入转换为DataFrame
    input_df = pd.DataFrame([features])
    # 进行预测
    prediction = model.predict_proba(scale.transform(input_df))[0][1]

    # 显示预测结果
    st.subheader('预测结果：')
    st.write(f'预测**概率: {prediction:.2%}')