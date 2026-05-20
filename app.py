# import pickle
# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Page config
# st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# # Custom CSS
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         background-color: #ff4b4b;
#         color: white;
#         border-radius: 10px;
#         height: 3em;
#         width: 100%;
#         font-size: 18px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
# st.markdown("---")

# # Load model
# model = pickle.load(open('gboost_model.pkl', 'rb'))

# # Sidebar Inputs
# st.sidebar.header("📝 Patient Details")

# Age = st.sidebar.slider('Age', 20, 100, 25)
# RestingBP = st.sidebar.slider('Resting BP', 0, 250, 100)
# Cholesterol = st.sidebar.slider('Cholesterol', 0, 650, 150)
# MaxHR = st.sidebar.slider('Max Heart Rate', 60, 250, 120)
# Oldpeak = st.sidebar.slider('Oldpeak', -3.0, 7.0, 1.0)

# FastingBS = st.sidebar.selectbox('Fasting Blood Sugar', (1, 0))
# gender = st.sidebar.selectbox('Gender', ('M', 'F'))
# ExerciseAngina = st.sidebar.selectbox('Exercise Angina', ('N', 'Y'))

# ChestPainType = st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
# RestingECG = st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
# ST_Slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

# # Encoding
# sex = 1 if gender == 'M' else 0
# exerciseAngina = 1 if ExerciseAngina == 'Y' else 0

# ChestPainType_ASY = 1 if ChestPainType == 'ASY' else 0
# ChestPainType_ATA = 1 if ChestPainType == 'ATA' else 0
# ChestPainType_NAP = 1 if ChestPainType == 'NAP' else 0
# ChestPainType_TA = 1 if ChestPainType == 'TA' else 0

# RestingECG_LVH = 1 if RestingECG == 'LVH' else 0
# RestingECG_Normal = 1 if RestingECG == 'Normal' else 0
# RestingECG_ST = 1 if RestingECG == 'ST' else 0

# st_Slope_dict = {'Up': 0, 'Down': 1, 'Flat': 2}
# st_Slope = st_Slope_dict[ST_Slope]

# # Create dataframe
# input_features = pd.DataFrame({
#     'Age': [Age],
#     'RestingBP': [RestingBP],
#     'Cholesterol': [Cholesterol],
#     'FastingBS': [FastingBS],
#     'MaxHR': [MaxHR],
#     'Oldpeak': [Oldpeak],
#     'sex': [sex],
#     'exerciseAngina': [exerciseAngina],
#     'RestingECG_LVH': [RestingECG_LVH],
#     'RestingECG_Normal': [RestingECG_Normal],
#     'RestingECG_ST': [RestingECG_ST],
#     'ChestPainType_ASY': [ChestPainType_ASY],
#     'ChestPainType_ATA': [ChestPainType_ATA],
#     'ChestPainType_NAP': [ChestPainType_NAP],
#     'ChestPainType_TA': [ChestPainType_TA],
#     'st_Slope': [st_Slope]
# })

# # Scaling
# scaler = StandardScaler()
# input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
#     input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
# )

# # Layout columns
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("📊 Input Summary")
#     st.dataframe(input_features)

# with col2:
#     st.subheader("⚡ Prediction")

#     if st.button("Predict Risk"):
#         prediction = model.predict(input_features)

#         if prediction[0] == 1:
#             st.error("🚨 High Risk of Heart Disease")
#         else:
#             st.success("✅ Low Risk of Heart Disease")

# # Optional visualization
# st.markdown("---")
# st.subheader("📈 Health Indicators")

# chart_data = pd.DataFrame({
#     'Feature': ['Age', 'BP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
#     'Value': [Age, RestingBP, Cholesterol, MaxHR, Oldpeak]
# })

# st.bar_chart(chart_data.set_index('Feature'))






import pickle
import streamlit as st
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #ff4b4b;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 30px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
    }

    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
    }

    .css-1d391kg {
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Based Healthcare Prediction Application</div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('gboost_model.pkl', 'rb'))

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📝 Patient Details")

Age = st.sidebar.slider('Age', 20, 100, 25)
RestingBP = st.sidebar.slider('Resting BP', 0, 250, 120)
Cholesterol = st.sidebar.slider('Cholesterol', 0, 650, 180)
MaxHR = st.sidebar.slider('Max Heart Rate', 60, 250, 150)
Oldpeak = st.sidebar.slider('Oldpeak', -3.0, 7.0, 1.0)

FastingBS = st.sidebar.selectbox('Fasting Blood Sugar', (1, 0))
gender = st.sidebar.selectbox('Gender', ('M', 'F'))
ExerciseAngina = st.sidebar.selectbox('Exercise Angina', ('N', 'Y'))

ChestPainType = st.sidebar.selectbox(
    'Chest Pain Type',
    ('ATA', 'NAP', 'ASY', 'TA')
)

RestingECG = st.sidebar.selectbox(
    'Resting ECG',
    ('Normal', 'ST', 'LVH')
)

ST_Slope = st.sidebar.selectbox(
    'ST Slope',
    ('Up', 'Flat', 'Down')
)

# =========================
# DISPLAY DATAFRAME
# =========================
display_df = pd.DataFrame({
    'Age': [Age],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'MaxHR': [MaxHR],
    'Oldpeak': [Oldpeak],
    'Gender': [gender],
    'ExerciseAngina': [ExerciseAngina],
    'ChestPainType': [ChestPainType],
    'RestingECG': [RestingECG],
    'ST_Slope': [ST_Slope]
})

# =========================
# ENCODING
# =========================
sex = 1 if gender == 'M' else 0
exerciseAngina = 1 if ExerciseAngina == 'Y' else 0

ChestPainType_ASY = 1 if ChestPainType == 'ASY' else 0
ChestPainType_ATA = 1 if ChestPainType == 'ATA' else 0
ChestPainType_NAP = 1 if ChestPainType == 'NAP' else 0
ChestPainType_TA = 1 if ChestPainType == 'TA' else 0

RestingECG_LVH = 1 if RestingECG == 'LVH' else 0
RestingECG_Normal = 1 if RestingECG == 'Normal' else 0
RestingECG_ST = 1 if RestingECG == 'ST' else 0

st_Slope_dict = {'Up': 0, 'Down': 1, 'Flat': 2}
st_Slope = st_Slope_dict[ST_Slope]

# =========================
# MODEL INPUT DATAFRAME
# =========================
input_features = pd.DataFrame({
    'Age': [Age],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'MaxHR': [MaxHR],
    'Oldpeak': [Oldpeak],
    'sex': [sex],
    'exerciseAngina': [exerciseAngina],
    'RestingECG_LVH': [RestingECG_LVH],
    'RestingECG_Normal': [RestingECG_Normal],
    'RestingECG_ST': [RestingECG_ST],
    'ChestPainType_ASY': [ChestPainType_ASY],
    'ChestPainType_ATA': [ChestPainType_ATA],
    'ChestPainType_NAP': [ChestPainType_NAP],
    'ChestPainType_TA': [ChestPainType_TA],
    'st_Slope': [st_Slope]
})

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([2, 1])

# =========================
# INPUT SUMMARY
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Input Summary")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION SECTION
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚡ Prediction")

    if st.button("Predict Risk"):

        prediction = model.predict(input_features)

        # Optional probability
        try:
            probability = model.predict_proba(input_features)[0][1]
            st.progress(int(probability * 100))
            st.write(f"### Risk Score: {round(probability * 100, 2)}%")
        except:
            pass

        if prediction[0] == 1:
            st.error("🚨 High Risk of Heart Disease")
            st.warning(
                "Suggestion: Consult a cardiologist and maintain a healthy lifestyle."
            )
        else:
            st.success("✅ Low Risk of Heart Disease")
            st.info(
                "Maintain healthy food habits and regular exercise."
            )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# VISUALIZATION
# =========================
st.markdown("---")

st.subheader("📈 Health Indicators")

chart_data = pd.DataFrame({
    'Feature': ['Age', 'BP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
    'Value': [Age, RestingBP, Cholesterol, MaxHR, Oldpeak]
})

st.bar_chart(
    chart_data.set_index('Feature')
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed using Machine Learning and Streamlit")


