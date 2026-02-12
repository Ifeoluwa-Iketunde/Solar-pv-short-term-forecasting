import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("PROCESSING FULL IRRADIANCE DATASET (2021-2023)")
print("=" * 60)

# Define paths
downloads_path = Path(os.path.expanduser("~/Downloads"))
dataset_path = downloads_path / "solar_dataset" / "extracted" / "Dataset" / "Time series dataset"

# Load irradiance datasets for all three years
print("\n1. Loading irradiance datasets for 2021, 2022, and 2023...")

irradiance_files = [
    dataset_path / "Meteorological dataset" / "Irradiance" / "Irradiance_2021.csv",
    dataset_path / "Meteorological dataset" / "Irradiance" / "Irradiance_2022.csv",
    dataset_path / "Meteorological dataset" / "Irradiance" / "Irradiance_2023.csv"
]

all_irradiance_data = []

for i, file_path in enumerate(irradiance_files):
    year = 2021 + i
    print(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"  {year} data: {df.shape[0]} observations, date range: {df['Time'].min()} to {df['Time'].max()}")
    all_irradiance_data.append(df)

# Concatenate all datasets
print("\n2. Concatenating all irradiance datasets...")
irradiance_df = pd.concat(all_irradiance_data, ignore_index=True)
print(f"Combined shape: {irradiance_df.shape}")

# Convert Time column to datetime and set as index
print("\n3. Converting Time column to datetime and setting index...")
irradiance_df['Time'] = pd.to_datetime(irradiance_df['Time'])
irradiance_df = irradiance_df.set_index('Time')

# Sort by Time
print("\n4. Sorting by Time...")
irradiance_df = irradiance_df.sort_index()
print(f"Sorted date range: {irradiance_df.index.min()} to {irradiance_df.index.max()}")

# Resample to hourly resolution using mean aggregation
print("\n5. Resampling to hourly resolution using mean aggregation...")
irradiance_hourly_full = irradiance_df.resample('h').mean()

print(f"Shape after resampling: {irradiance_hourly_full.shape}")
print(f"Date range: {irradiance_hourly_full.index.min()} to {irradiance_hourly_full.index.max()}")

# Remove rows with missing values
print("\n6. Removing rows with missing values...")
rows_before = irradiance_hourly_full.shape[0]
irradiance_hourly_full = irradiance_hourly_full.dropna()
rows_after = irradiance_hourly_full.shape[0]
print(f"Rows removed: {rows_before - rows_after}")
print(f"Rows remaining: {rows_after}")

# Save the combined dataset
print("\n7. Saving combined irradiance dataset...")
output_path = Path("data/irradiance_hourly_full.csv")
irradiance_hourly_full.to_csv(output_path)
print(f"Saved combined irradiance hourly data to: {output_path}")

# Display summary statistics
print(f"\n8. Summary Statistics:")
print(f"Total observations: {irradiance_hourly_full.shape[0]}")
print(f"Date range: {irradiance_hourly_full.index.min()} to {irradiance_hourly_full.index.max()}")
print(f"Years covered: {sorted(irradiance_hourly_full.index.year.unique())}")

print(f"\nIrradiance descriptive statistics:")
print(irradiance_hourly_full['Irradiance (W/m2)'].describe())

# Show year-by-year breakdown
print(f"\nYear-by-year breakdown:")
for year in sorted(irradiance_hourly_full.index.year.unique()):
    year_data = irradiance_hourly_full[irradiance_hourly_full.index.year == year]
    print(f"  {year}: {year_data.shape[0]} observations")
    print(f"    Mean Irradiance: {year_data['Irradiance (W/m2)'].mean():.2f} W/m²")
    print(f"    Max Irradiance: {year_data['Irradiance (W/m2)'].max():.2f} W/m²")

print("\n" + "=" * 60)
print("FULL IRRADIANCE DATASET PROCESSING COMPLETE!")
print("=" * 60)
print(f"\nFile created: data/irradiance_hourly_full.csv")
print(f"Total size: {irradiance_hourly_full.shape[0]} hourly observations")
print(f"Coverage: {irradiance_hourly_full.index.min().strftime('%Y-%m-%d')} to {irradiance_hourly_full.index.max().strftime('%Y-%m-%d')}")