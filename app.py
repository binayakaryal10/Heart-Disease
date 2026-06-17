import pandas as pd
import streamlit as st
import joblib 

# -------------------------------------------------------------------
# 1. Page Configuration & Loading
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    # Using caching so artifacts don't reload on every user interaction
    model = joblib.load("Svm_heard.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Please ensure all .pkl files are in the directory.")
    st.stop()

# -------------------------------------------------------------------
# 2. Header Section
# -------------------------------------------------------------------
st.title("Heart Disease Risk Predictor")
st.markdown(
    """
    Provide the clinical metrics below to estimate the likelihood of heart disease. 
    Please ensure all values are accurate before clicking **Analyze Risk**.
    """
)
st.write("---")

# -------------------------------------------------------------------
# 3. Form Layout & User Inputs
# -------------------------------------------------------------------
with st.form(key="prediction_form"):
    
    st.subheader("📋 Patient Demographics & Vitals")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 40, help="Age of the patient in years")
        sex = st.selectbox("Sex", ['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", 
            options=[0, 1], 
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
        )
        chest_pain = st.selectbox(
            "Chest Pain Type", 
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic"
        )

    st.write("---")
    st.subheader("🫀 Electrocardiogram & Stress Test Metrics")
    col3, col4 = st.columns(2)

    with col3:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["Y", "N"],
            help="Does exercise cause angina (chest pain)?"
        )

    with col4:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope Type", ["Up", "Flat", "Down"])

    # Form submit button prevents app rerunning on every single slider change
    submit_button = st.form_submit_button(label="⚡ Analyze Risk", use_container_width=True)

# -------------------------------------------------------------------
# 4. Processing & Prediction
# -------------------------------------------------------------------
if submit_button:
    with st.spinner("Processing clinical data..."):
        # Initialize raw input dict with 0s for missing dummy columns upfront
        raw_input = {col: 0 for col in expected_columns}
        
        # Populate the known numerical and base fields
        raw_input['Age'] = age
        raw_input['RestingBP'] = resting_bp
        raw_input['Cholesterol'] = cholesterol
        raw_input['FastingBS'] = fasting_bs
        raw_input['MaxHR'] = max_hr
        raw_input['Oldpeak'] = oldpeak
        
        # Set dynamic dummy variables to 1
        raw_input['Sex_' + sex] = 1
        raw_input['ChestPainType_' + chest_pain] = 1
        raw_input['RestingECG_' + resting_ecg] = 1
        raw_input['ExerciseAngina_' + exercise_angina] = 1
        raw_input['ST_Slope_' + st_slope] = 1

        # Create DataFrame ensuring perfect column order match
        input_df = pd.DataFrame([raw_input])[expected_columns]

        # Scale features
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

    # -------------------------------------------------------------------
    # 5. Professional Results Display
    # -------------------------------------------------------------------
    st.write("---")
    st.subheader("📊 Diagnostic Results")
    
    if prediction == 1:
        st.error("### ⚠️ High Risk of Heart Disease Detected")
        st.markdown(
            "The model indicates clinical patterns strongly associated with cardiovascular issues. "
            "Please consult a medical professional for a comprehensive evaluation."
        )
    else:
        st.success("### ✅ Low Risk of Heart Disease Detected")
        st.markdown(
            "The model indicates clinical metrics within normal ranges. "
            "Continue maintaining a healthy lifestyle and regular checkups!"
        )