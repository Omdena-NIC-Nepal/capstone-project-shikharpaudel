import pandas as pd

def engineer_climate_features(df):
    df['temp_avg'] = (df['max_temp_2m'] + df['min_temp_2m']) / 2
    df['humidity_diff'] = df['relative_humidity_2m'] - df['specific_humidity_2m']
    df['pressure_temp_ratio'] = df['surface_pressure'] / df['air_temp_2m']
    df['wind_10m_variation'] = df['max_wind_speed_10m'] - df['min_wind_speed_10m']
    df['wind_50m_variation'] = df['max_wind_speed_50m'] - df['min_wind_speed_50m']
    df['temp_fluctuation'] = df['max_temp_2m'] - df['min_temp_2m']
    return df
