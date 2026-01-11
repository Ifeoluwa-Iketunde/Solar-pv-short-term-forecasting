import pandas as pd

# Load Plant 1 Generation Data with datetime parsing
generation_df = pd.read_csv('data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'])

# Load Plant 1 Weather Sensor Data with datetime parsing
weather_df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'])

# Display Generation Data information
print("=" * 60)
print("PLANT 1 GENERATION DATA")
print("=" * 60)
print("\nColumn Names and Data Types:")
print(generation_df.dtypes)
print("\nDataset Shape:", generation_df.shape)
print("\nFirst 5 Rows:")
print(generation_df.head())

# Display Weather Sensor Data information
print("\n" + "=" * 60)
print("PLANT 1 WEATHER SENSOR DATA")
print("=" * 60)
print("\nColumn Names and Data Types:")
print(weather_df.dtypes)
print("\nDataset Shape:", weather_df.shape)
print("\nFirst 5 Rows:")
print(weather_df.head())
