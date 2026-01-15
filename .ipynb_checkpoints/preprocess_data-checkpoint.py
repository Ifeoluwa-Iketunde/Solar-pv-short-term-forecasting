import pandas as pd

print("=" * 60)
print("SOLAR PV DATA PREPROCESSING")
print("=" * 60)

# Load Plant 1 Generation Data with datetime parsing
print("\n1. Loading data...")
generation_df = pd.read_csv('data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'])
weather_df = pd.read_csv('data/Plant_2_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'])

print(f"Generation Data Shape: {generation_df.shape}")
print(f"Weather Data Shape: {weather_df.shape}")

# Merge dataframes on DATE_TIME using inner join
print("\n2. Merging datasets...")
print(f"Shape before merging:")
print(f"  Generation Data: {generation_df.shape}")
print(f"  Weather Data: {weather_df.shape}")

merged_df = pd.merge(generation_df, weather_df, on='DATE_TIME', how='inner')
merged_df = merged_df.sort_values('DATE_TIME').reset_index(drop=True)

print(f"Duplicate timestamps before removal: {merged_df.duplicated(subset='DATE_TIME').sum()}")
merged_df = merged_df.drop_duplicates(subset='DATE_TIME', keep='first')

print(f"Shape after merging: {merged_df.shape}")
print(f"Duplicate timestamps after removal: {merged_df.duplicated(subset='DATE_TIME').sum()}")

# Resample to hourly resolution
print("\n3. Resampling to hourly resolution...")
merged_df['DATE_TIME'] = pd.to_datetime(merged_df['DATE_TIME'])
merged_df = merged_df.set_index('DATE_TIME')

# Select only numeric columns for resampling
numeric_cols = merged_df.select_dtypes(include=['number']).columns
merged_df_numeric = merged_df[numeric_cols]

resampled_df = merged_df_numeric.resample('h').mean()
rows_before = resampled_df.shape[0]

# Remove rows where IRRADIATION == 0
resampled_df = resampled_df[resampled_df['IRRADIATION'] != 0]

rows_after = resampled_df.shape[0]
rows_removed = rows_before - rows_after
print(f"Rows removed (IRRADIATION == 0): {rows_removed}")
print(f"Rows remaining: {rows_after}")

# Handle missing values
print("\n4. Handling missing values...")
print("Missing Values Percentage:")
missing_pct = (resampled_df.isnull().sum() / len(resampled_df)) * 100
print(missing_pct)

print(f"\nShape before cleaning: {resampled_df.shape}")

# Identify weather columns
weather_cols = [col for col in resampled_df.columns if col not in ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]

# Apply linear interpolation to weather variables
resampled_df[weather_cols] = resampled_df[weather_cols].interpolate(method='linear')

# Drop rows with missing DC_POWER values
resampled_df = resampled_df.dropna(subset=['DC_POWER'])

print(f"Shape after cleaning: {resampled_df.shape}")

# Save cleaned dataset
print("\n5. Saving cleaned dataset...")
resampled_df.to_csv('data/solar_pv_clean_hourly.csv')

print(f"\nFinal dataset shape: {resampled_df.shape}")
print(f"Date range: {resampled_df.index.min()} to {resampled_df.index.max()}")
print(f"\nDataset saved to: data/solar_pv_clean_hourly.csv")

# Sanity check
print("\n" + "=" * 60)
print("SANITY CHECK")
print("=" * 60)

print("\nColumn Names:")
print(resampled_df.columns.tolist())

print(f"\nDate Range:")
print(f"  Start: {resampled_df.index.min()}")
print(f"  End: {resampled_df.index.max()}")

print(f"\nDC_POWER Descriptive Statistics:")
print(resampled_df['DC_POWER'].describe())

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
