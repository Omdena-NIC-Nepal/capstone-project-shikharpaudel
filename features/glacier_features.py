import pandas as pd

def engineer_glacier_features(df):
    df['elevation_range'] = df['Maximum Elevation'] - df['Minimum Elevation']
    df['area_per_length'] = df['Glacier Area'] / df['Mean Length']
    df['area_per_depth'] = df['Glacier Area'] / df['Mean Depth']
    df['elevation_to_slope_ratio'] = df['Mean Elevation'] / df['Average Slope (deg)']
    return df
