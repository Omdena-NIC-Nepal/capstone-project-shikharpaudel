import streamlit as st
import pandas as pd

def show_prediction():
    st.title("Climate Feature Prediction")

    # Define the input features
    year = st.number_input("Select Year", min_value=1980, max_value=2030, value=2023)
    month = st.selectbox("Select Month", list(range(1, 13)))
    province_names = ["1", "2", "3", "4", "5", "6", "7"]  # Change if province names differ
    province_input = st.selectbox("Select Province", province_names)

    # Check if encoder and models are in session_state
    if "province_label_encoder" not in st.session_state:
        st.error("Province label encoder not found in memory. Please train the models first.")
        return

    if "trained_models" not in st.session_state:
        st.error("Trained models not found in memory. Please train the models first.")
        return

    #  Get encoder from memory
    label_encoder = st.session_state.province_label_encoder

    try:
        province_encoded = label_encoder.transform([province_input])[0]
    except Exception as e:
        st.error(f"Province encoding error: {e}")
        return

    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'province': [province_encoded]
    })

    # Target climate features
    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    #  Predict using cached models
    predictions = {}
    for target in targets:
        if target not in st.session_state.trained_models:
            st.warning(f"Model for '{target}' not found in session.")
            continue

        model = st.session_state.trained_models[target]
        pred = model.predict(input_data)[0]
        predictions[target] = round(pred, 2)

    if predictions:
        st.subheader("Predicted Climate Features")
        st.table(pd.DataFrame([predictions]))
    else:
        st.warning("No predictions could be made. Please ensure models are trained.")
