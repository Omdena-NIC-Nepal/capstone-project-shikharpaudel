import streamlit as st
import pandas as pd

def show_prediction():
    st.title("Climate Feature Prediction")

    # Check if models and encoder exist in memory
    if "trained_models" not in st.session_state or "label_encoder" not in st.session_state:
        st.error("Please train the models first in the Model Training section.")
        return

    # User input
    year = st.number_input("Select Year", min_value=1980, max_value=2030, value=2023)
    month = st.selectbox("Select Month", list(range(1, 13)))
    province_name = st.selectbox("Select Province", st.session_state.label_encoder.classes_)

    try:
        province_encoded = st.session_state.label_encoder.transform([province_name])[0]
    except Exception as e:
        st.error(f"Province encoding failed: {e}")
        return

    input_df = pd.DataFrame({
        'year': [year],
        'month': [month],
        'province': [province_encoded]
    })

    predictions = {}
    for target, model in st.session_state.trained_models.items():
        try:
            pred = model.predict(input_df)[0]
            predictions[target] = round(pred, 2)
        except Exception as e:
            st.warning(f"Prediction failed for {target}: {e}")

    if predictions:
        st.subheader("Predicted Climate Features")
        st.table(pd.DataFrame([predictions]))
    else:
        st.warning("❌ No predictions available. Train models first.")
