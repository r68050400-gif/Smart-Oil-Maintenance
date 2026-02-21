import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load trained model
model = joblib.load("model/predictive_model.pkl")

st.set_page_config(page_title="Smart Oil Maintenance", layout="centered")

st.title("🛢 Smart Oil Maintenance System")
st.subheader("Enter Equipment Sensor Values")

# Input sliders
temperature = st.slider("Temperature", 40, 120, 75)
pressure = st.slider("Pressure", 10, 60, 30)
vibration = st.slider("Vibration", 1, 10, 5)
runtime = st.slider("Runtime Hours", 500, 4000, 2000)

if st.button("Check Risk"):

    input_data = np.array([[temperature, pressure, vibration, runtime]])

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = round(probability * 100, 2)

    st.write(f"## Risk Percentage: {risk_percent}%")

    # Dynamic Gauge Color
    if risk_percent < 30:
        gauge_color = "green"
    elif risk_percent < 70:
        gauge_color = "yellow"
    else:
        gauge_color = "red"

    # Risk Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 30], 'color': '#2ecc71'},
                {'range': [30, 70], 'color': '#f1c40f'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ]
        }
    ))

    st.plotly_chart(fig)

    # Alert System
    if risk_percent < 30:
        st.success("✅ Equipment is Safe")
    elif risk_percent < 70:
        st.warning("⚠ Warning: Maintenance Recommended")
    else:
        st.error("🚨 Critical Risk! Immediate Action Needed")

    # Cost Estimation
    downtime_cost = risk_percent * 1000
    st.write(f"### Estimated Downtime Cost: ₹{int(downtime_cost)}")

    # Failure Reason Detection
    reasons = []

    if temperature > 85:
        reasons.append("High Temperature")

    if vibration > 6.5:
        reasons.append("High Vibration")

    if pressure > 40:
        reasons.append("High Pressure")

    if reasons:
        st.write("### Possible Failure Reasons:")
        for r in reasons:
            st.write(f"- {r}")

    # Feature Importance
    st.write("### Key Risk Factors Importance")

    importance = model.feature_importances_
    features = ["Temperature", "Pressure", "Vibration", "Runtime Hours"]

    for i in range(len(features)):
        st.write(f"{features[i]}: {round(importance[i]*100,2)}%")
