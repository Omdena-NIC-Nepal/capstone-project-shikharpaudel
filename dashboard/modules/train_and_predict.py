import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def show_train_and_predict():
    st.title("Model Training and Prediction")

    # Load data
    data_path = "data/featured/feature_climate.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("Climate feature file not found.")
        return

    if st.checkbox("Show raw climate feature data"):
        st.dataframe(df.head())

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode 'province' and save label encoder in session state
    if "label_encoder" not in st.session_state:
        label_encoder = LabelEncoder()
        df['province'] = label_encoder.fit_transform(df['province'])
        st.session_state.label_encoder = label_encoder
    else:
        label_encoder = st.session_state.label_encoder
        df['province'] = label_encoder.transform(df['province'])

    # Features and Targets
    features = ['year', 'month', 'province']
    targets = [
        'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
        'max_temp_2m', 'min_temp_2m', 'wind_speed_10m',
        'max_wind_speed_10m', 'min_wind_speed_10m'
    ]

    test_size = st.slider("Select test size", 0.1, 0.5, 0.2, 0.05)

    if st.button("Train Models"):
        st.write("Training models...")
        st.session_state.trained_models = {}
        st.session_state.models_trained = False

        for target in targets:
            if target not in df.columns:
                st.warning(f"{target} not found in data.")
                continue

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            st.session_state.trained_models[target] = model

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            st.success(f"✅ Trained model for `{target}`")
            st.markdown(f"- **MSE**: {mse:.2f}")
            st.markdown(f"- **RMSE**: {rmse:.2f}")
            st.markdown(f"- **R² Score**: {r2:.2f}")

        st.session_state.models_trained = True
        st.success("🎉 All models trained and stored in memory.")

    # --- Prediction Section ---
    st.markdown("---")
    st.subheader("📈 Make Predictions")

    if not st.session_state.get("models_trained", False):
        st.info("⚠️ Please train the models above before making predictions.")
        return

    # Prediction input
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
        st.subheader("🔮 Predicted Climate Features")
        st.table(pd.DataFrame([predictions]))
    else:
        st.warning("❌ No predictions available.")
