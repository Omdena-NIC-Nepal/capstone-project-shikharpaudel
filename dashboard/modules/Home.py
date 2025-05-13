import streamlit as st

def show_home():
    st.title("🌍 Climate Change Impact Dashboard – Nepal")
    st.subheader("📘 Omdena Batch II Capstone Project")

    st.markdown("""
    Welcome to the **Climate Change Impact Dashboard**!  
    This tool is designed to analyze and predict the effects of climate change in Nepal using real climate and glacier data combined with machine learning models.
    """)

    st.markdown("## 📌 Dashboard Sections Overview")

    st.markdown("""
    ### 🏠 Home  
    You're here! This is your starting point — learn what each section does and how to use the dashboard step-by-step.

    ### 📈 Climate Trends  
    Visualize yearly climate trends like **temperature, humidity, wind, and precipitation** across districts and provinces using raw climate data.

    ### 🧊 Glacier Overview  
    View glacier distributions on an interactive map. Explore glacier **elevation**, **area**, and **geographic patterns** across Nepal.

    ### 🔍 Exploratory Data Analysis (EDA)  
    Analyze climate features using various **visualizations**, uncovering patterns and correlations for deeper understanding.

    ### 🛠️ Feature Engineering  
    Transform raw data into **engineered features** that improve machine learning model performance.

    ### 🤖 Model Training  
    Select and train machine learning models on featured climate data. Evaluate model performance with key metrics.

    ### 🔮 Prediction  
    Enter **year, month, and province** to predict:  
    - 🌧️ Precipitation  
    - 🌡️ Average, Max, Min Temperature  
    - 🌫️ Relative Humidity  
    - 🌬️ Wind Speeds (10m and extremes)

    Results are shown in a clean vertical table.

    ### 💬 NLP & Sentiment Analysis  
    Type in any **climate-related text** (e.g., news headlines, local observations).  
    The app analyzes the **sentiment** (Positive, Neutral, Negative) and displays it in a **bar chart**.
    """)

    st.markdown("## 🧭 How to Navigate")
    st.markdown("""
    Use the **sidebar menu** to switch between different sections of the dashboard.

    👉 Start from **Climate Trends** → proceed through each stage to build understanding, train models, and make predictions!

    Enjoy exploring the climate of Nepal through data! 🌿
    """)

