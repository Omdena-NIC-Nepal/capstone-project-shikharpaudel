import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def run_eda(climate_df,glacier_df):
    st.header("📊 Exploratory Data Analysis")

    st.header("1. Data Preview")
    st.subheader("Climate Data Overview")
    st.dataframe(climate_df.head())
    st.subheader("Glacier Data Overview")
    st.dataframe(glacier_df.head())

   
    st.subheader(f"2. Air Temperature Trend in Different Districts.")
    district = st.selectbox("Select District", sorted(climate_df["district"].unique()))
    df_district = climate_df[climate_df['district'] == district]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_district, x='year', y='air_temp_2m')
    plt.title(f"Air Temperature Trend in {district}")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("3. Precipitation Distribution by Province")
    q1 = climate_df['precipitation_total'].quantile(0.25)
    q3 = climate_df['precipitation_total'].quantile(0.75)
    IQR = q3 - q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR
    filtered_df = climate_df[(climate_df['precipitation_total'] >= lower) & 
                             (climate_df['precipitation_total'] <= upper)]

    plt.figure(figsize=(12, 10))
    sns.boxplot(data=filtered_df, x="province", y="precipitation_total")
    plt.title("Precipitation Distribution By Province")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("4. Correlation Matrix")
    exclude_col = ['province','latitude','longitude']
    numeric_cols = climate_df.select_dtypes(include='number').columns
    desired_cols = [col for col in numeric_cols if col not in exclude_col]
    corr = climate_df[desired_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Climate Variables")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("5. Wind Speed (10m) Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(climate_df['wind_speed_10m'], bins=30, kde=True)
    plt.title("Distribution of Wind Speed (10m)")
    st.pyplot(plt.gcf())
    plt.clf()

    # AVerage Air Temperature Per Year
    st.subheader("6. Average Air Temperature Per Year")
    avg_temp = climate_df.groupby("year")["air_temp_2m"].mean()
    plt.figure(figsize = (10,8))
    avg_temp.plot(title="Average Air Temperature Per Year",marker = 'o', linestyle = '-',linewidth = 2, markersize = 8)
    plt.xlabel("Year")
    plt.ylabel("Average Temperature in Degree Celcius")
    st.pyplot(plt.gcf())
    plt.clf()    

    # average monthly precipitation trend
    st.subheader("7. Average Monthly Precipitation Trend")
    avg_preci = climate_df.groupby("month")["precipitation_total"].mean()

    plt.figure(figsize=(10, 6))
    avg_preci.plot(kind='bar', title="Average Monthly Precipitation")
    plt.xlabel("Month")
    plt.ylabel("Average Precipitation (mm)")
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
           rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    # temperature by province
    st.subheader("8. Temperature by Province")
    plt.figure(figsize = (10,8))
    sns.boxplot(x = 'province', y = 'air_temp_2m',data = climate_df)
    plt.xlabel("Province")
    plt.ylabel("Temperature")
    st.pyplot(plt.gcf())
    plt.clf()

    # Air Temperature vs Humidity
    st.subheader("9. Air Temperature vs Humidity")

    if climate_df.empty or 'air_temp_2m' not in climate_df.columns or 'relative_humidity_2m' not in climate_df.columns:
      st.warning("Data is empty or required columns are missing.")
    else:
      fig, ax = plt.subplots(figsize=(10, 8))
      sampled_df = climate_df.sample(n=min(1000, len(climate_df)), random_state=42)
      st.text(f"Plotting {len(sampled_df)} points after sampling")
    
      sns.scatterplot(x='air_temp_2m', y='relative_humidity_2m', data=sampled_df, alpha=0.6, s=50, ax=ax)
      sns.regplot(x='air_temp_2m', y='relative_humidity_2m', data=climate_df, scatter=False, color='red', lowess=True, ax=ax)

      ax.set_title("Air Temperature vs. Relative Humidity", fontsize=14, pad=10)
      ax.set_xlabel("Air Temperature (2m, °C)", fontsize=12)
      ax.set_ylabel("Relative Humidity (2m, %)", fontsize=12)
      ax.grid(True, linestyle='--', alpha=0.7)

      st.pyplot(fig)  # Use the 'fig' directly here
      plt.clf()
 

    # wind speed vs humidity
    st.subheader("10. Wind Speed vs Humidity")
    fig, ax = plt.subplots(figsize=(10, 8))
    sampled_data = climate_df.sample(n=min(1000, len(climate_df)), random_state=42)
    print(f"Plotting {len(sampled_data)} points after sampling")
    sns.scatterplot(x='wind_speed_10m', y='air_temp_2m', data=sampled_data, alpha = 0.6, s = 50)
    #  regression line to show trend
    sns.regplot(x='wind_speed_10m', y='air_temp_2m', data=sampled_data, 
            scatter=False, color='red', lowess=True)  # LOWESS fit for non-linear trend
    ax.set_xlabel("Wind Speed")  # Use ax.set_xlabel instead of plt.xlabel
    ax.set_ylabel("Air Temperature")
    ax.set_title("Wind Speed vs Air Temperature")

    st.pyplot(plt.gcf())
    plt.clf()  

    # distribution of surface skin temperature
    st.subheader("11. Distribution of Surface Skin Temperature")
    fig, ax  = plt.subplots(figsize = (10,6))
    sns.histplot(climate_df["surface_skin_temp"], bins=30, kde=True)
    ax.set_title("Distribution of Surface Skin Temperature")
    ax.set_xlabel("Surface Skin Temperature")
    ax.set_ylabel("Temperature Count")
    st.pyplot(plt.gcf())
    plt.clf()     

    #Wind Speed Range (10m and 50m) Distribution
    st.subheader("12. Wind Speed Range (10m and 50m) Distribution")
    fig, ax = plt.subplots(figsize = (10,6))

    sns.boxplot(data=climate_df[["wind_speed_range_10m", "wind_speed_range_50m"]])
    ax.set_title("Wind Speed Range Comparison at 10m and 50m")
    st.pyplot(plt.gcf())
    plt.clf() 

    #Histogram of glaciar data
    st.subheader("13. Histogram of Glacier Data")
    fig, ax = plt.subplots(figsize = (10,6))
    sns.histplot(glacier_df['Glacier Area'], bins=30, kde=True)
    plt.title("Distribution of Glacier Area")
    ax.set_xlabel("Area (sq. km)")
    ax.set_ylabel("Number of Glaciers")
    st.pyplot(plt.gcf())
    plt.clf() 

    # Glacier area vs mean elevation
    st.subheader("14. Glacier Area vs Mean Elevation")
    fig = px.scatter(glacier_df, 
                 x='Mean Elevation', 
                 y='Glacier Area', 
                 color='Primary Class', 
                 hover_name = 'Glacier Source',
    )
    # Display the plot
    st.plotly_chart(fig)

    
