import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write(
    "Enter passenger and service details below to predict whether the passenger is "
    "**Satisfied** or **Neutral/Dissatisfied**."
)

st.divider()

# -----------------------------
# Input form
# -----------------------------
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer", "disloyal Customer"]
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

        travel_type = st.selectbox(
            "Type of Travel",
            ["Business travel", "Personal Travel"]
        )

        travel_class = st.selectbox(
            "Class",
            ["Business", "Eco", "Eco Plus"]
        )

        flight_distance = st.number_input(
            "Flight Distance",
            min_value=0,
            value=500
        )

        wifi = st.slider(
            "Inflight Wifi Service",
            0, 5, 3
        )

        dep_arr_convenient = st.slider(
            "Departure/Arrival Time Convenient",
            0, 5, 3
        )

    with col2:
        online_booking = st.slider(
            "Ease of Online Booking",
            0, 5, 3
        )

        gate_location = st.slider(
            "Gate Location",
            0, 5, 3
        )

        food_drink = st.slider(
            "Food and Drink",
            0, 5, 3
        )

        online_boarding = st.slider(
            "Online Boarding",
            0, 5, 3
        )

        seat_comfort = st.slider(
            "Seat Comfort",
            0, 5, 3
        )

        entertainment = st.slider(
            "Inflight Entertainment",
            0, 5, 3
        )

        onboard_service = st.slider(
            "On-board Service",
            0, 5, 3
        )

        leg_room = st.slider(
            "Leg Room Service",
            0, 5, 3
        )

    with col3:
        baggage = st.slider(
            "Baggage Handling",
            0, 5, 3
        )

        checkin = st.slider(
            "Check-in Service",
            0, 5, 3
        )

        inflight_service = st.slider(
            "Inflight Service",
            0, 5, 3
        )

        cleanliness = st.slider(
            "Cleanliness",
            0, 5, 3
        )

        dep_delay = st.number_input(
            "Departure Delay in Minutes",
            min_value=0,
            value=0
        )

        arr_delay = st.number_input(
            "Arrival Delay in Minutes",
            min_value=0,
            value=0
        )

    submit = st.form_submit_button("Predict Satisfaction")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Customer Type": [customer_type],
        "Age": [age],
        "Type of Travel": [travel_type],
        "Class": [travel_class],
        "Flight Distance": [flight_distance],
        "Inflight wifi service": [wifi],
        "Departure/Arrival time convenient": [dep_arr_convenient],
        "Ease of Online booking": [online_booking],
        "Gate location": [gate_location],
        "Food and drink": [food_drink],
        "Online boarding": [online_boarding],
        "Seat comfort": [seat_comfort],
        "Inflight entertainment": [entertainment],
        "On-board service": [onboard_service],
        "Leg room service": [leg_room],
        "Baggage handling": [baggage],
        "Checkin service": [checkin],
        "Inflight service": [inflight_service],
        "Cleanliness": [cleanliness],
        "Departure Delay in Minutes": [dep_delay],
        "Arrival Delay in Minutes": [arr_delay]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ Prediction: SATISFIED")
        st.metric(
            "Satisfaction Probability",
            f"{probability[1] * 100:.2f}%"
        )
    else:
        st.error("❌ Prediction: NEUTRAL OR DISSATISFIED")
        st.metric(
            "Dissatisfaction Probability",
            f"{probability[0] * 100:.2f}%"
        )

    with st.expander("View Input Data"):
        st.dataframe(input_data)