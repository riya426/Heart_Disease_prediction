import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('heart_disease_model.pkl')

# Streamlit UI
st.title("💖 Heart Disease Prediction System")

# User input fields
st.sidebar.header("Enter Patient Details:")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure(bps)", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol(chol) (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar(fbs) > 120 mg/dL (1 = True, 0 = False)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved(thalach)", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina(exang) (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise(oldspeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Convert input data to array
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("✅ The person *does not* have heart disease.")
    else:
        st.error("⚠️ The person *has* heart disease.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by our Team Members(Aditi , Aniket , Mangaldip , Riya , Shivani)")