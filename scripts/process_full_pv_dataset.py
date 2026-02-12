import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("PROCESSING FULL PV DATASET (2021-2023)")
print("=" * 60)

# Define paths
downloads_path = Path(os.path.expanduser("~/Downloads"))
dataset_path = downloads_path / "solar_dataset" / "extracted" / "Dataset" / "Time series dataset"

# Load the full PV generation dataset for SQ1 station
print("\n1. Loading PV generation data for SQ1 station (2021-2023)...")
pv_file_path = dataset_path / "PV generation dataset" / "PV stations with panel level optimizer" / "Site level dataset" / "SQ1.csv"

print(f"Loading file: {pv_file_path}")
pv_df = pd.read_csv(pv_file_path)

print(f"Original shape: {pv_df.shape}")
print(f"Columns: {pv_df.columns.tolist()}")
print(f"Date range: {pv_df['Time'].min()} to {pv_df['Time'].max()}")

# Convert Time column to datetime
print("\n2. Converting Time column to datetime format...")
pv_df['Time'] = pd.to_datetime(pv_df['Time'])
pv_df = pv_df.set_index('Time')

# Sort by Time
print("\n3. Sorting by Time...")
pv_df = pv_df.sort_index()

print(f"Sorted date range: {pv_df.index.min()} to {pv_df.index.max()}")
print(f"Years present: {sorted(pv_df.index.year.unique())}")

# Keep only the Time and generation(kWh) columns, rename generation(kWh) to Solar_Energy
print("\n4. Selecting and renaming columns...")
pv_df = pv_df[['generation(kWh)']].copy()
pv_df = pv_df.rename(columns={'generation(kWh)': 'Solar_Energy'})

print(f"Shape after column selection: {pv_df.shape}")

# Resample to hourly resolution using sum aggregation
print("\n5. Resampling to hourly resolution using sum aggregation...")
pv_hourly_full = pv_df.resample('h').sum()

print(f"Shape after resampling: {pv_hourly_full.shape}")
print(f"Date range: {pv_hourly_full.index.min()} to {pv_hourly_full.index.max()}")

# Remove rows with missing values
print("\n6. Removing rows with missing values...")
rows_before = pv_hourly_full.shape[0]
pv_hourly_full = pv_hourly_full.dropna()
rows_after = pv_hourly_full.shape[0]
print(f"Rows removed: {rows_before - rows_after}")
print(f"Rows remaining: {rows_after}")

# Save the combined dataset
print("\n7. Saving combined dataset...")
output_path = Path("data/pv_hourly_full.csv")
pv_hourly_full.to_csv(output_path)
print(f"Saved combined PV hourly data to: {output_path}")

# Display summary statistics
print(f"\n8. Summary Statistics:")
print(f"Total observations: {pv_hourly_full.shape[0]}")
print(f"Date range: {pv_hourly_full.index.min()} to {pv_hourly_full.index.max()}")
print(f"Years covered: {sorted(pv_hourly_full.index.year.unique())}")

print(f"\nSolar_Energy descriptive statistics:")
print(pv_hourly_full['Solar_Energy'].describe())

# Show year-by-year breakdown
print(f"\nYear-by-year breakdown:")
for year in sorted(pv_hourly_full.index.year.unique()):
    year_data = pv_hourly_full[pv_hourly_full.index.year == year]
    print(f"  {year}: {year_data.shape[0]} observations")
    print(f"    Mean Solar_Energy: {year_data['Solar_Energy'].mean():.2f} kWh")
    print(f"    Max Solar_Energy: {year_data['Solar_Energy'].max():.2f} kWh")

print("\n" + "=" * 60)
print("FULL PV DATASET PROCESSING COMPLETE!")
print("=" * 60)
print(f"\nFile created: data/pv_hourly_full.csv")
print(f"Total size: {pv_hourly_full.shape[0]} hourly observations")
print(f"Coverage: {pv_hourly_full.index.min().strftime('%Y-%m-%d')} to {pv_hourly_full.index.max().strftime('%Y-%m-%d')}")