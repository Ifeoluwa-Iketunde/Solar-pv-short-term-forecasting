import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=" * 60)
print("RANDOM FOREST REGRESSOR IMPLEMENTATION")
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

# Set Time as index
train_data = train_data.set_index('Time')
test_data = test_data.set_index('Time')

print(f"Training date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Testing date range: {test_data.index.min()} to {test_data.index.max()}")

# Prepare features and target
print("\n3. Preparing features and target variables...")
feature_columns = ['Irradiance (W/m2)', 'Temperature', 'Wind']
target_column = 'Solar_Energy'

X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Testing target shape: {y_test.shape}")

print(f"\nFeature columns: {feature_columns}")
print(f"Target column: {target_column}")

# Display feature statistics
print(f"\nTraining Data Statistics:")
print("Features:")
print(X_train.describe())
print("\nTarget:")
print(y_train.describe())

# Train Random Forest Regressor
print("\n4. Training Random Forest Regressor...")
print("   Features: Irradiance, Temperature, Wind")
print("   Target: Solar_Energy")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Fitting model...")
rf_model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions
print("\n5. Making predictions on test set...")
y_pred = rf_model.predict(X_test)

print(f"Predictions shape: {y_pred.shape}")
print(f"Sample predictions: {y_pred[:5]}")

# Calculate metrics
print("\n6. Calculating evaluation metrics...")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Display results
print("\n" + "=" * 60)
print("RANDOM FOREST MODEL RESULTS")
print("=" * 60)

print(f"\nEvaluation Metrics (Test Set):")
print(f"  MAE (Mean Absolute Error): {mae:.4f} kWh")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} kWh")
print(f"  R² (R-squared): {r2:.4f}")

print(f"\nInterpretation:")
print(f"  MAE: On average, predictions are off by {mae:.4f} kWh")
print(f"  RMSE: Larger errors are penalized more heavily ({rmse:.4f} kWh)")
if r2 > 0:
    print(f"  R²: Model explains {r2*100:.1f}% of the variance in solar energy generation")
else:
    print(f"  R²: Model performs worse than simply predicting the mean ({r2:.4f})")

# Feature Importance
print(f"\nFeature Importance:")
print(feature_importance)

# Model Performance Analysis
print(f"\nModel Performance Analysis:")
print(f"  Best performing feature: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.3f})")
print(f"  Worst performing feature: {feature_importance.iloc[-1]['Feature']} ({feature_importance.iloc[-1]['Importance']:.3f})")

# Show sample predictions vs actual
print(f"\nSample Predictions vs Actual (first 10):")
comparison_df = pd.DataFrame({
    'Actual': y_test.head(10),
    'Predicted': y_pred[:10],
    'Error': (y_test.head(10) - y_pred[:10])
})
print(comparison_df)

# Compare with persistence baseline
print(f"\nComparison with Persistence Baseline:")
print(f"  Random Forest MAE: {mae:.4f} kWh")
print(f"  Random Forest RMSE: {rmse:.4f} kWh")
print(f"  Random Forest R²: {r2:.4f}")

# Performance improvement calculation (if persistence results are available)
try:
    # Load persistence results for comparison
    persistence_mae = 1.4402
    persistence_rmse = 2.6638
    persistence_r2 = 0.8242
    
    mae_improvement = (persistence_mae - mae) / persistence_mae * 100
    rmse_improvement = (persistence_rmse - rmse) / persistence_rmse * 100
    r2_improvement = (r2 - persistence_r2) / persistence_r2 * 100
    
    print(f"\nPerformance Improvement vs Persistence Model:")
    print(f"  MAE Improvement: {mae_improvement:+.1f}%")
    print(f"  RMSE Improvement: {rmse_improvement:+.1f}%")
    print(f"  R² Improvement: {r2_improvement:+.1f}%")
    
    if mae_improvement > 0:
        print(f"  ✓ Random Forest improves MAE by {mae_improvement:.1f}%")
    else:
        print(f"  ✗ Random Forest performs worse on MAE by {abs(mae_improvement):.1f}%")
        
    if rmse_improvement > 0:
        print(f"  ✓ Random Forest improves RMSE by {rmse_improvement:.1f}%")
    else:
        print(f"  ✗ Random Forest performs worse on RMSE by {abs(rmse_improvement):.1f}%")
        
    if r2_improvement > 0:
        print(f"  ✓ Random Forest improves R² by {r2_improvement:.1f}%")
    else:
        print(f"  ✗ Random Forest performs worse on R² by {abs(r2_improvement):.1f}%")
        
except:
    print(f"  (Comparison with persistence baseline not available)")

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nKey Findings:")
print(f"  - Model trained on {len(feature_columns)} meteorological features")
print(f"  - Most important feature: {feature_importance.iloc[0]['Feature']}")
print(f"  - Test set performance: R² = {r2:.4f}")
print(f"\nNext steps:")
print(f"  - Try feature engineering (time-based features)")
print(f"  - Experiment with hyperparameter tuning")
print(f"  - Consider LSTM for sequential patterns")