import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("PERSISTENCE MODEL IMPLEMENTATION")
print("=" * 60)

# Load train and test datasets
print("\n1. Loading datasets...")
train_data = pd.read_csv('data/train_data_full.csv')
test_data = pd.read_csv('data/test_data_full.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Convert Time column to datetime
print("\n2. Converting Time column to datetime...")
train_data['Time'] = pd.to_datetime(train_data['Time'])
test_data['Time'] = pd.to_datetime(test_data['Time'])

# Set Time as index for easier handling
train_data = train_data.set_index('Time')
test_data = test_data.set_index('Time')

print(f"Training date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Testing date range: {test_data.index.min()} to {test_data.index.max()}")

# Implement persistence model
print("\n3. Implementing persistence model...")
print("   Forecast = Previous hour's Solar_Energy value")

# For the test set, the persistence forecast is the previous hour's actual value
# We need to shift the actual values by 1 to create the persistence forecast
test_actual = test_data['Solar_Energy']
test_persistence_forecast = test_actual.shift(1)  # Previous hour's value

# Remove the first row since we don't have a previous value for it
test_actual = test_actual.iloc[1:]
test_persistence_forecast = test_persistence_forecast.iloc[1:]

print(f"Test data points for evaluation: {len(test_actual)}")

# Calculate metrics
print("\n4. Calculating evaluation metrics...")

mae = mean_absolute_error(test_actual, test_persistence_forecast)
rmse = np.sqrt(mean_squared_error(test_actual, test_persistence_forecast))
r2 = r2_score(test_actual, test_persistence_forecast)

# Display results
print("\n" + "=" * 60)
print("PERSISTENCE MODEL RESULTS")
print("=" * 60)

print(f"\nEvaluation Metrics (Test Set):")
print(f"  MAE (Mean Absolute Error): {mae:.4f} kWh")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} kWh")
print(f"  R² (R-squared): {r2:.4f}")

print(f"\nInterpretation:")
print(f"  MAE: On average, predictions are off by {mae:.4f} kWh")
print(f"  RMSE: Larger errors are penalized more heavily ({rmse:.4f} kWh)")
if r2 > 0:
    print(f"  R²: Model explains {r2*100:.1f}% of the variance in the data")
else:
    print(f"  R²: Model performs worse than simply predicting the mean ({r2:.4f})")

# Show sample predictions
print(f"\nSample Predictions (first 10):")
comparison_df = pd.DataFrame({
    'Actual': test_actual.head(10),
    'Persistence_Forecast': test_persistence_forecast.head(10),
    'Error': (test_actual - test_persistence_forecast).head(10)
})
print(comparison_df)

# Show when persistence works well vs poorly
print(f"\nWhen Persistence Model Works Well:")
print(f"  - During periods of stable weather")
print(f"  - When solar generation changes gradually")
print(f"  - During consistent daily patterns")

print(f"\nWhen Persistence Model Struggles:")
print(f"  - During rapid weather changes")
print(f"  - At sunrise/sunset transitions")
print(f"  - During cloud cover events")

print("\n" + "=" * 60)
print("PERSISTENCE MODEL EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nThis serves as your baseline model for comparison")
print(f"with more sophisticated forecasting approaches.")