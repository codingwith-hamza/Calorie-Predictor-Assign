import pickle

import pandas as pd
import streamlit as st

MODEL_FILE = "calories_model_full_dataset.pkl"

st.set_page_config(page_title="Calories Burnt Predictor", page_icon="🔥")


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("🔥 Calories Burnt Prediction System")
st.write(
    "Enter your details and workout measurements, then click **Predict**. "
    "The model was trained on 15,000 workout records and compared using "
    "10-fold cross validation."
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age (years)", min_value=20, max_value=79, value=30, step=1)
    height = st.number_input("Height (cm)", min_value=123.0, max_value=222.0, value=175.0, step=1.0)
    weight = st.number_input("Weight (kg)", min_value=36.0, max_value=132.0, value=75.0, step=1.0)

with col2:
    duration = st.number_input("Workout duration (minutes)", min_value=1.0, max_value=30.0, value=20.0, step=1.0)
    heart_rate = st.number_input("Average heart rate (beats/min)", min_value=67.0, max_value=128.0, value=100.0, step=1.0)
    body_temp = st.number_input("Body temperature (°C)", min_value=37.1, max_value=41.5, value=40.0, step=0.1)

if st.button("Predict", type="primary"):
    # Same columns, in the same order, as the training data
    person = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender": gender,
    }])
    calories = model.predict(person)[0]
    st.success(f"Estimated calories burnt: **{calories:,.1f} kcal**")
    st.caption("Typical model error (cross-validation RMSE): about 0.44 kcal.")

st.divider()
st.caption("Developed by Ameer Hamza Aslam | COMSATS University Lahore")
