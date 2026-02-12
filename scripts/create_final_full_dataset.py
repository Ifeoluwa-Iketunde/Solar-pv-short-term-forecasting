import pandas as pd

print("=" * 60)
print("CREATING FINAL FULL SOLAR FORECASTING DATASET")
print("=" * 60)

# Load all processed full datasets
print("\n1. Loading all processed full datasets...")

pv_data = pd.read_csv('data/pv_hourly_full.csv', index_col='Time', parse_dates=True)
irradiance_data = pd.read_csv('data/irradiance_hourly_full.csv', index_col='Time', parse_dates=True)
temperature_data = pd.read_csv('data/temperature_hourly_full.csv', index_col='Time', parse_dates=True)
wind_data = pd.read_csv('data/wind_hourly_full.csv', index_col='Time', parse_dates=True)

print(f"PV Data shape: {pv_data.shape}")
print(f"Irradiance Data shape: {irradiance_data.shape}")
print(f"Temperature Data shape: {temperature_data.shape}")
print(f"Wind Data shape: {wind_data.shape}")

# Check date ranges
print(f"\nPV Data range: {pv_data.index.min()} to {pv_data.index.max()}")
print(f"Irradiance Data range: {irradiance_data.index.min()} to {irradiance_data.index.max()}")
print(f"Temperature Data range: {temperature_data.index.min()} to {temperature_data.index.max()}")
print(f"Wind Data range: {wind_data.index.min()} to {wind_data.index.max()}")

# Merge all datasets on Time column using inner join
print("\n2. Merging all datasets using inner join...")
final_df = pv_data.join([irradiance_data, temperature_data, wind_data], how='inner')

# Sort by Time
print("\n3. Sorting by Time...")
final_df = final_df.sort_index()

print(f"Shape after merging: {final_df.shape}")
print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

# Remove any rows with missing values
print("\n4. Removing rows with missing values...")
rows_before = final_df.shape[0]
final_df = final_df.dropna()
rows_after = final_df.shape[0]
print(f"Rows removed: {rows_before - rows_after}")
print(f"Rows remaining: {rows_after}")

# Save the final dataset
print("\n5. Saving final dataset...")
final_df.to_csv('data/solar_forecasting_full_dataset.csv')
print(f"Saved final dataset to: data/solar_forecasting_full_dataset.csv")

# Display final dataset info
print(f"\n6. Final dataset information:")
print(f"Columns: {final_df.columns.tolist()}")
print(f"Years covered: {sorted(final_df.index.year.unique())}")

print(f"\nDescriptive Statistics:")
print(final_df.describe())

# Show year-by-year breakdown
print(f"\nYear-by-year breakdown:")
for year in sorted(final_df.index.year.unique()):
    year_data = final_df[final_df.index.year == year]
    print(f"  {year}: {year_data.shape[0]} observations")

# Show correlation analysis
print(f"\nCorrelation Analysis:")
correlation_matrix = final_df.corr()
print("Correlation with Solar_Energy:")
print(correlation_matrix[['Solar_Energy']].sort_values('Solar_Energy', ascending=False))

print("\n" + "=" * 60)
print("FINAL DATASET CREATION COMPLETE!")
print("=" * 60)
print(f"\nFile created: data/solar_forecasting_full_dataset.csv")
print(f"Total observations: {final_df.shape[0]}")
print(f"Features: {len(final_df.columns)}")
print(f"Date range: {final_df.index.min().strftime('%Y-%m-%d')} to {final_df.index.max().strftime('%Y-%m-%d')}")
print(f"Years covered: {sorted(final_df.index.year.unique())}")
print(f"\nReady for temporal train/test split!")