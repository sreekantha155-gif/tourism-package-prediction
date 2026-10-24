from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
META_PATH = HERE / "model_metadata.json"

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️", layout="centered")
st.title("Wellness Tourism Package Prediction")
st.write("Estimate whether a customer is likely to purchase the tourism package. This is a decision-support estimate, not a guarantee.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run the GitHub Actions pipeline first so it commits model.joblib.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
feature_names = metadata.get("features", [])

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        product = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    with c2:
        duration = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=10)
        visitors = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
        property_star = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
        trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=2)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
        own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Children Visiting (below age 5)", min_value=0, max_value=10, value=0)
        income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=20000)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    row = {
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": visitors, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": property_star,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income
    }
    input_df = pd.DataFrame([row])
    # Align columns to the model's training feature order.
    if feature_names:
        input_df = input_df.reindex(columns=feature_names)
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction == 1:
        st.success("Model prediction: likely to purchase.")
    else:
        st.info("Model prediction: less likely to purchase.")
    st.caption("Use this score to prioritize outreach; validate against business outcomes.")
