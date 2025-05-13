# modules/feature_engineering_view.py

import streamlit as st
import pandas as pd

def show_feature_engineering():
    st.title("🔧 Feature Engineering Results")

    # Load engineered data
    try:
        climate_df = pd.read_csv("data/featured/feature_climate.csv")
        glacier_df = pd.read_csv("data/featured/feature_glacier.csv")
    except FileNotFoundError:
        st.error("Engineered feature files not found. Please run feature engineering first.")
        return

   #Climate Data 
    st.header("🌡️ Climate Data (Engineered Features)")
    st.dataframe(climate_df.head())

    st.subheader("📈 Average Temperature by Year")
    temp_by_year = climate_df.groupby("year")["temp_avg"].mean()
    st.line_chart(temp_by_year)

    st.subheader("💨 Wind Speed Variation (10m) by Year")
    wind_var = climate_df.groupby("year")["wind_10m_variation"].mean()
    st.line_chart(wind_var)

    # Glacier Data 
    st.header("🗻 Glacier Data (Engineered Features)")
    st.dataframe(glacier_df.head())

    st.subheader("⛰️ Elevation Range of Top 10 Glaciers")
    st.bar_chart(glacier_df["elevation_range"].head(10))

    st.subheader("📏 Area per Length of Top 10 Glaciers")
    st.bar_chart(glacier_df["area_per_length"].head(10))
