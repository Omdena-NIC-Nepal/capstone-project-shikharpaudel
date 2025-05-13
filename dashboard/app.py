import streamlit as st
import pandas as pd
import plotly.express as px
import os
from modules.Home import show_home
from modules.eda import run_eda
from modules.featured_view import show_feature_engineering
from modules.train_climate_model_view import show_model_training
from modules.predict_view import show_prediction
from modules.footer import show_footer

#page config
st.set_page_config(page_title = "Climate Impact in Nepal", layout = "wide")

#load data
@st.cache_data
def load_data():
    climate = pd.read_csv("data/raw/climate_data_nepal_district_wise_monthly_province_grouped.csv")
    glacier = pd.read_csv("data/raw/nepal_glacier_data.csv")
    return climate, glacier
climate_df, glacier_df = load_data()
# sidebar Navigation
page = st.sidebar.radio("select a Page", ["Home","Climate Trends","Glacier Overview","EDA","Feature Engineering","Model Training and Prediction", "NLP Sentiment Analysis"])
# Home Page
if page == "Home":
    show_home()
# Climate Trends Page
# Climate Trends Page
elif page == "Climate Trends":
    st.header("📈 Climate Trends by Districts")

    # Dropdown for climate variable selection
    climate_var_options = {
        "Precipitation": "precipitation_total",
        "Air Temperature (2m)": "air_temp_2m",
        "Max Temperature (2m)": "max_temp_2m",
        "Min Temperature (2m)": "min_temp_2m",
        "Temperature Range (2m)": "temp_range_2m",
        "Surface Skin Temperature": "surface_skin_temp",
        "Relative Humidity (2m)": "relative_humidity_2m",
        "Specific Humidity (2m)": "specific_humidity_2m",
        "Wet Bulb Temperature (2m)": "wet_bulb_temp_2m",
        "Surface Pressure": "surface_pressure",
        "Wind Speed (10m)": "wind_speed_10m",
        "Max Wind Speed (10m)": "max_wind_speed_10m",
        "Wind Speed Range (10m)": "wind_speed_range_10m",
        "Wind Speed (50m)": "wind_speed_50m",
        "Wind Speed Range (50m)": "wind_speed_range_50m",
    }

    climate_variable_label = st.selectbox("Select climate variable", list(climate_var_options.keys()))
    climate_variable = climate_var_options[climate_variable_label]

    # District selection (ensure filtered_df and column exist)
    available_districts = climate_df['district'].unique()
    selected_district = st.selectbox("Select a district", sorted(available_districts))

    # Filter the DataFrame for the selected district
    filtered_df = climate_df[climate_df['district'] == selected_district]

    # Plot only if data is available
    if not filtered_df.empty:
        fig = px.line(
            filtered_df,
            x="date",
            y=climate_variable,
            title=f"{climate_variable_label} Trend in {selected_district}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {selected_district}.")


# Glacier Overview Page
elif page == "Glacier Overview":
    st.header(" Glacier Distribution and Elevation")

    fig = px.scatter_mapbox(
        glacier_df,
        lat="Latitude",
        lon="Longitude",
        color="Mean Elevation",
        size="Mean Elevation",  # corrected here
        mapbox_style="open-street-map",
        zoom=5,
        height = 700,
        width = 900,
        title="Glacier Locations in Nepal (by Mean Elevation)"
    )
    st.plotly_chart(fig, use_container_width=True)
# eda page
elif page == "EDA":
    run_eda(climate_df,glacier_df)
    
elif page == "Feature Engineering":
    show_feature_engineering()

# model training page
elif page == "Model Training and Prediction":
     from modules.train_and_predict import show_train_and_predict
     show_train_and_predict()
# Simple Prediction Page
#elif page == "Prediction":
   # st.header("📊 Climate Prediction (Mockup)")
   # st.markdown("This page can show simple regression model results.")
   # show_prediction()

elif page == "NLP Sentiment Analysis":
    st.header("📝 NLP Sentiment Analysis")
    from modules.NLP_sentiment_view import show_sentiment_analysis
    show_sentiment_analysis()

# footer show
show_footer()