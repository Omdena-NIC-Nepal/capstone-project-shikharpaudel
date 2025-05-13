import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def show_model_training():
    st.title("Model Training - Climate Prediction")

    # Load featured climate dataset
    data_path = "data/featured/feature_climate.csv"
    if not os.path.exists(data_path):
        st.error("Climate feature file not found.")
        return

    df = pd.read_csv(data_path)

    # Display raw data option
    if st.checkbox("Show raw climate feature data"):
        st.dataframe(df.head())

    # Clean the data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode 'province'
    if df['province'].dtype == 'object':
        label_encoder = LabelEncoder()
        df['province'] = label_encoder.fit_transform(df['province'])
        os.makedirs("models", exist_ok=True)
        joblib.dump(label_encoder, "models/province_label_encoder.pkl")
    else:
        label_encoder = None

    features = ['year', 'month', 'province']
    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    if st.button("Train Models"):
        st.write("Training models...")
        for target in targets:
            if target not in df.columns:
                st.warning(f"{target} not found in data.")
                continue

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Save model
            joblib.dump(model, f"models/{target}_model.pkl")

            st.success(f"Trained model for {target}")
            st.markdown(f"- **MSE**: {mse:.2f}")
            st.markdown(f"- **RMSE**: {rmse:.2f}")
            st.markdown(f"- **R² Score**: {r2:.2f}")
