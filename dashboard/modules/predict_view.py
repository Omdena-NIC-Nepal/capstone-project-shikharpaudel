import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def show_prediction():
    st.title("Climate Feature Prediction")

    # Define the input features
    year = st.number_input("Select Year", min_value=1980, max_value=2030, value=2023)
    month = st.selectbox("Select Month", list(range(1, 13)))
    province_names = [
        "1", "2", "3", "4", "5", "6", "7"
    ]  # Update this if provinces are named differently in your data
    province_input = st.selectbox("Select Province", province_names)

    model_dir = "models"
    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    # Load Label Encoder
    encoder_path = os.path.join(model_dir, "province_label_encoder.pkl")
    if not os.path.exists(encoder_path):
        st.error("Label encoder not found. Please train the model first.")
        return

    label_encoder = joblib.load(encoder_path)
    
    try:
        province_encoded = label_encoder.transform([province_input])[0]
    except:
        st.error("Invalid province selection. Please match with training data.")
        return

    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'province': [province_encoded]
    })

    predictions = {}
    for target in targets:
        model_path = os.path.join(model_dir, f"{target}_model.pkl")
        if not os.path.exists(model_path):
            st.warning(f"Model for {target} not found.")
            continue

        model = joblib.load(model_path)
        pred = model.predict(input_data)[0]
        predictions[target] = round(pred, 2)

    if predictions:
        st.subheader("Predicted Climate Features")
        result_df = pd.DataFrame([predictions])
        st.table(result_df)
    else:
        st.warning("No predictions could be made. Please check the models.")
