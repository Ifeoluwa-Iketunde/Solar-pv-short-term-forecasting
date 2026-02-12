import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("PROCESSING FULL WIND DATASET (2021-2023)")
print("=" * 60)

# Define paths
downloads_path = Path(os.path.expanduser("~/Downloads"))
dataset_path = downloads_path / "solar_dataset" / "extracted" / "Dataset" / "Time series dataset"

# Load wind datasets for all three years
print("\n1. Loading wind datasets for 2021, 2022, and 2023...")

wind_files = [
    dataset_path / "Meteorological dataset" / "Wind" / "Wind_2021.csv",
    dataset_path / "Meteorological dataset" / "Wind" / "Wind_2022.csv",
    dataset_path / "Meteorological dataset" / "Wind" / "Wind_2023.csv"
]

all_wind_data = []

for i, file_path in enumerate(wind_files):
    year = 2021 + i
    print(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"  {year} data: {df.shape[0]} observations, date range: {df['Time'].min()} to {df['Time'].max()}")
    all_wind_data.append(df)

# Concatenate all datasets
print("\n2. Concatenating all wind datasets...")
wind_df = pd.concat(all_wind_data, ignore_index=True)
print(f"Combined shape: {wind_df.shape}")

# Convert Time column to datetime and set as index
print("\n3. Converting Time column to datetime and setting index...")
wind_df['Time'] = pd.to_datetime(wind_df['Time'])
wind_df = wind_df.set_index('Time')

# Sort by Time
print("\n4. Sorting by Time...")
wind_df = wind_df.sort_index()
print(f"Sorted date range: {wind_df.index.min()} to {wind_df.index.max()}")

# Resample to hourly resolution using mean aggregation
print("\n5. Resampling to hourly resolution using mean aggregation...")
wind_hourly_full = wind_df.resample('h').mean()

print(f"Shape after resampling: {wind_hourly_full.shape}")
print(f"Date range: {wind_hourly_full.index.min()} to {wind_hourly_full.index.max()}")

# Remove rows with missing values
print("\n6. Removing rows with missing values...")
rows_before = wind_hourly_full.shape[0]
wind_hourly_full = wind_hourly_full.dropna()
rows_after = wind_hourly_full.shape[0]
print(f"Rows removed: {rows_before - rows_after}")
print(f"Rows remaining: {rows_after}")

# Rename column to Wind (using Wind Speed column)
print("\n7. Renaming column to Wind...")
wind_hourly_full = wind_hourly_full.rename(columns={'Wind Speed (m/s)': 'Wind'})

# Save the combined dataset
print("\n8. Saving combined wind dataset...")
output_path = Path("data/wind_hourly_full.csv")
wind_hourly_full.to_csv(output_path)
print(f"Saved combined wind hourly data to: {output_path}")

# Display summary statistics
print(f"\n9. Summary Statistics:")
print(f"Total observations: {wind_hourly_full.shape[0]}")
print(f"Date range: {wind_hourly_full.index.min()} to {wind_hourly_full.index.max()}")
print(f"Years covered: {sorted(wind_hourly_full.index.year.unique())}")

print(f"\nWind descriptive statistics:")
print(wind_hourly_full['Wind'].describe())

# Show year-by-year breakdown
print(f"\nYear-by-year breakdown:")
for year in sorted(wind_hourly_full.index.year.unique()):
    year_data = wind_hourly_full[wind_hourly_full.index.year == year]
    print(f"  {year}: {year_data.shape[0]} observations")
    print(f"    Mean Wind Speed: {year_data['Wind'].mean():.2f} m/s")
    print(f"    Max Wind Speed: {year_data['Wind'].max():.2f} m/s")

print("\n" + "=" * 60)
print("FULL WIND DATASET PROCESSING COMPLETE!")
print("=" * 60)
print(f"\nFile created: data/wind_hourly_full.csv")
print(f"Total size: {wind_hourly_full.shape[0]} hourly observations")
print(f"Coverage: {wind_hourly_full.index.min().strftime('%Y-%m-%d')} to {wind_hourly_full.index.max().strftime('%Y-%m-%d')}")