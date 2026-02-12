import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("PROCESSING FULL TEMPERATURE DATASET (2021-2023)")
print("=" * 60)

# Define paths
downloads_path = Path(os.path.expanduser("~/Downloads"))
dataset_path = downloads_path / "solar_dataset" / "extracted" / "Dataset" / "Time series dataset"

# Load temperature datasets for all three years
print("\n1. Loading temperature datasets for 2021, 2022, and 2023...")

temperature_files = [
    dataset_path / "Meteorological dataset" / "Temperature" / "Temperature_2021.csv",
    dataset_path / "Meteorological dataset" / "Temperature" / "Temperature_2022.csv",
    dataset_path / "Meteorological dataset" / "Temperature" / "Temperature_2023.csv"
]

all_temperature_data = []

for i, file_path in enumerate(temperature_files):
    year = 2021 + i
    print(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"  {year} data: {df.shape[0]} observations, date range: {df['Time'].min()} to {df['Time'].max()}")
    all_temperature_data.append(df)

# Concatenate all datasets
print("\n2. Concatenating all temperature datasets...")
temperature_df = pd.concat(all_temperature_data, ignore_index=True)
print(f"Combined shape: {temperature_df.shape}")

# Convert Time column to datetime and set as index
print("\n3. Converting Time column to datetime and setting index...")
temperature_df['Time'] = pd.to_datetime(temperature_df['Time'])
temperature_df = temperature_df.set_index('Time')

# Sort by Time
print("\n4. Sorting by Time...")
temperature_df = temperature_df.sort_index()
print(f"Sorted date range: {temperature_df.index.min()} to {temperature_df.index.max()}")

# Resample to hourly resolution using mean aggregation
print("\n5. Resampling to hourly resolution using mean aggregation...")
temperature_hourly_full = temperature_df.resample('h').mean()

print(f"Shape after resampling: {temperature_hourly_full.shape}")
print(f"Date range: {temperature_hourly_full.index.min()} to {temperature_hourly_full.index.max()}")

# Remove rows with missing values
print("\n6. Removing rows with missing values...")
rows_before = temperature_hourly_full.shape[0]
temperature_hourly_full = temperature_hourly_full.dropna()
rows_after = temperature_hourly_full.shape[0]
print(f"Rows removed: {rows_before - rows_after}")
print(f"Rows remaining: {rows_after}")

# Rename column to Temperature
print("\n7. Renaming column to Temperature...")
temperature_hourly_full = temperature_hourly_full.rename(columns={'Temp (Degree Celsius)': 'Temperature'})

# Save the combined dataset
print("\n8. Saving combined temperature dataset...")
output_path = Path("data/temperature_hourly_full.csv")
temperature_hourly_full.to_csv(output_path)
print(f"Saved combined temperature hourly data to: {output_path}")

# Display summary statistics
print(f"\n9. Summary Statistics:")
print(f"Total observations: {temperature_hourly_full.shape[0]}")
print(f"Date range: {temperature_hourly_full.index.min()} to {temperature_hourly_full.index.max()}")
print(f"Years covered: {sorted(temperature_hourly_full.index.year.unique())}")

print(f"\nTemperature descriptive statistics:")
print(temperature_hourly_full['Temperature'].describe())

# Show year-by-year breakdown
print(f"\nYear-by-year breakdown:")
for year in sorted(temperature_hourly_full.index.year.unique()):
    year_data = temperature_hourly_full[temperature_hourly_full.index.year == year]
    print(f"  {year}: {year_data.shape[0]} observations")
    print(f"    Mean Temperature: {year_data['Temperature'].mean():.2f} °C")
    print(f"    Max Temperature: {year_data['Temperature'].max():.2f} °C")

print("\n" + "=" * 60)
print("FULL TEMPERATURE DATASET PROCESSING COMPLETE!")
print("=" * 60)
print(f"\nFile created: data/temperature_hourly_full.csv")
print(f"Total size: {temperature_hourly_full.shape[0]} hourly observations")
print(f"Coverage: {temperature_hourly_full.index.min().strftime('%Y-%m-%d')} to {temperature_hourly_full.index.max().strftime('%Y-%m-%d')}")