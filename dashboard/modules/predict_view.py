import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

def show_prediction():
    st.title("Climate Feature Prediction")

    # User inputs
    year = st.number_input("Select Year", min_value=1980, max_value=2030, value=2023)
    month = st.selectbox("Select Month", list(range(1, 13)))
    province_names = ["1", "2", "3", "4", "5", "6", "7"]  # Update if needed
    province_input = st.selectbox("Select Province", province_names)

    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    # ✅ Load label encoder from disk only when needed
    encoder_path = "models/province_label_encoder.pkl"
    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
    else:
        st.error("Province label encoder not found. Please train the models first.")
        return

    # ✅ Encode province
    try:
        province_encoded = label_encoder.transform([province_input])[0]
    except Exception as e:
        st.error(f"Province encoding error: {str(e)}")
        return

    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'province': [province_encoded]
    })

    predictions = {}

    # ✅ Load model from disk only when needed
    for target in targets:
        model_path = f"models/{target}_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.warning(f"Model for {target} not found.")
            continue

        # Make predictions
        pred = model.predict(input_data)[0]
        predictions[target] = round(pred, 2)

    if predictions:
        st.subheader("Predicted Climate Features")
        st.table(pd.DataFrame([predictions]))
    else:
        st.warning("No predictions could be made. Please check the models.")
