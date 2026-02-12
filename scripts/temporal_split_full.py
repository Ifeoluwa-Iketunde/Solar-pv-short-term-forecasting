import pandas as pd
from datetime import datetime

print("=" * 60)
print("CREATING TEMPORAL TRAIN/TEST SPLIT (2021-2022 vs 2023)")
print("=" * 60)

# Load the full solar forecasting dataset
print("\n1. Loading solar_forecasting_full_dataset.csv...")
df = pd.read_csv('data/solar_forecasting_full_dataset.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Convert Time column to datetime
print("\n2. Converting Time column to datetime format...")
df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time')

# Sort the dataset by Time
print("\n3. Sorting dataset by Time...")
df = df.sort_index()

print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Years present: {sorted(df.index.year.unique())}")

# Split the data based on time periods
print("\n4. Creating temporal split...")

# Training data: from 2021-01-01 to 2022-12-31
train_start = datetime(2021, 1, 1)
train_end = datetime(2022, 12, 31)
train_data = df[(df.index >= train_start) & (df.index <= train_end)]

# Testing data: from 2023-01-01 to 2023-12-31
test_start = datetime(2023, 1, 1)
test_end = datetime(2023, 12, 31)
test_data = df[(df.index >= test_start) & (df.index <= test_end)]

print(f"\nTraining data (2021-2022):")
print(f"  Date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"  Shape: {train_data.shape}")
print(f"  Percentage of total data: {train_data.shape[0]/df.shape[0]*100:.1f}%")
print(f"  Years: {sorted(train_data.index.year.unique())}")

print(f"\nTesting data (2023):")
print(f"  Date range: {test_data.index.min()} to {test_data.index.max()}")
print(f"  Shape: {test_data.shape}")
print(f"  Percentage of total data: {test_data.shape[0]/df.shape[0]*100:.1f}%")
print(f"  Years: {sorted(test_data.index.year.unique())}")

# Save the training dataset
print("\n5. Saving training dataset...")
train_data.to_csv('data/train_data_full.csv')
print(f"Training data saved to: data/train_data_full.csv")

# Save the testing dataset
print("\n6. Saving testing dataset...")
test_data.to_csv('data/test_data_full.csv')
print(f"Testing data saved to: data/test_data_full.csv")

# Display summary statistics for both datasets
print("\n" + "=" * 60)
print("TEMPORAL SPLIT SUMMARY")
print("=" * 60)

print(f"\nTraining Data (2021-2022) Statistics:")
print(f"  Rows: {train_data.shape[0]}")
print(f"  Columns: {train_data.shape[1]}")
print(f"  Features: {train_data.columns.tolist()}")
print(train_data.describe())

print(f"\n\nTesting Data (2023) Statistics:")
print(f"  Rows: {test_data.shape[0]}")
print(f"  Columns: {test_data.shape[1]}")
print(f"  Features: {test_data.columns.tolist()}")
print(test_data.describe())

print(f"\n" + "=" * 60)
print("TEMPORAL SPLIT COMPLETE!")
print("=" * 60)
print(f"\nFiles created:")
print(f"  - data/train_data_full.csv ({train_data.shape[0]} rows)")
print(f"  - data/test_data_full.csv ({test_data.shape[0]} rows)")
print(f"\nTemporal split ratio: {train_data.shape[0]}:{test_data.shape[0]} ({train_data.shape[0]/(train_data.shape[0]+test_data.shape[0])*100:.1f}% training)")
print(f"\nReady for model training and evaluation!")