import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

# 1) Load the full pipeline (preprocess + model)
model = joblib.load('heartsense_pipeline.pkl')

st.title("💓 HeartSense - Heart Disease Risk Predictor")

st.sidebar.header("Enter Your Health Details")

def user_input():
    age = st.sidebar.slider('Age', 20, 80, 45)
    sex = st.sidebar.selectbox('Sex (1 = male, 0 = female)', [1, 0])
    cp = st.sidebar.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting BP', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar >120', [1, 0])
    restecg = st.sidebar.selectbox('Resting ECG (0–2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise-Induced Angina', [1, 0])
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope (0–2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Major Vessels (0–4)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thal (1,2,3)', [1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

# 2) Collect + show input (raw schema, no manual get_dummies here)
input_df = user_input()
st.subheader("Your Input:")
st.write(input_df)

# 3) Predict using the pipeline directly
prediction = model.predict(input_df)                       # 0/1
risk_score = model.predict_proba(input_df)[0][1] * 100     # % probability

# 4) Risk banding for user-friendly messaging
def categorize_risk(score):
    if score < 30:
        return "Low Risk"
    elif 30 <= score < 60:
        return "Moderate Risk"
    else:
        return "High Risk"

risk_level = categorize_risk(risk_score)

st.subheader("Prediction Result")
if risk_level == "Low Risk":
    st.success("✅ You are at Low Risk of heart disease.")
    st.info("Keep maintaining a healthy lifestyle with regular exercise and balanced diet.")
elif risk_level == "Moderate Risk":
    st.warning("⚠️ You are at Moderate Risk of heart disease.")
    st.info("Consider consulting a doctor and adopting preventive measures.")
else:
    st.error("🚨 You are at High Risk of heart disease.")
    st.warning("It is strongly advised to consult a healthcare professional immediately.")

st.write(f"Risk Score: **{risk_score:.2f}%**")
st.write(f"Risk Level: **{risk_level}**")

# === The rest of your visuals (gauges, ranges, report download) can stay exactly as you had ===
# They use the raw input_df (good), while the model uses its internal pipeline (also good).
