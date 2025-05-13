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

    # Define paths
    data_path = "data/featured/feature_climate.csv"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists

    # Load dataset
    if not os.path.exists(data_path):
        st.error("❌ Climate feature file not found.")
        return

    df = pd.read_csv(data_path)

    if st.checkbox("Show raw climate feature data"):
        st.dataframe(df.head())

    # Clean and validate data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode 'province'
    if df['province'].dtype == 'object':
        try:
            label_encoder = LabelEncoder()
            df['province'] = label_encoder.fit_transform(df['province'])
            joblib.dump(label_encoder, os.path.join(models_dir, "province_label_encoder.pkl"))
        except Exception as e:
            st.error(f"Label encoding failed: {e}")
            return

    features = ['year', 'month', 'province']
    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    if st.button("Train Models"):
        st.info("⏳ Training models... Please wait.")

        for target in targets:
            if target not in df.columns:
                st.warning(f"⚠️ {target} not found in data. Skipping.")
                continue

            try:
                # Prepare training data
                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Save model
                model_filename = os.path.join(models_dir, f"{target}_model.pkl")
                joblib.dump(model, model_filename)

                # Show results
                st.success(f"✅ Trained model for: {target}")
                st.markdown(f"- **MSE**: {mse:.2f}")
                st.markdown(f"- **RMSE**: {rmse:.2f}")
                st.markdown(f"- **R² Score**: {r2:.2f}")

            except Exception as e:
                st.error(f"❌ Error training model for {target}: {e}")
