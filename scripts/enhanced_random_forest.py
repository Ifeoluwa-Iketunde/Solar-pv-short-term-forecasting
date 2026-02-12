import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=" * 60)
print("ENHANCED RANDOM FOREST REGRESSOR")
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

# Create enhanced features
print("\n3. Creating enhanced temporal and lag features...")

def create_features(df):
    """Create enhanced features for better forecasting"""
    df_features = df.copy()
    
    # Temporal features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_year'] = df_features.index.dayofyear
    df_features['month'] = df_features.index.month
    
    # Cyclical encoding for hour (captures day/night cycles)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Cyclical encoding for day of year (captures seasonal patterns)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    # Lag features (previous hour's values)
    df_features['solar_lag1'] = df_features['Solar_Energy'].shift(1)
    df_features['irradiance_lag1'] = df_features['Irradiance (W/m2)'].shift(1)
    df_features['temperature_lag1'] = df_features['Temperature'].shift(1)
    df_features['wind_lag1'] = df_features['Wind'].shift(1)
    
    # 2-hour lag features
    df_features['solar_lag2'] = df_features['Solar_Energy'].shift(2)
    df_features['irradiance_lag2'] = df_features['Irradiance (W/m2)'].shift(2)
    
    # Rolling features (recent averages)
    df_features['solar_rolling_mean_3'] = df_features['Solar_Energy'].rolling(window=3, min_periods=1).mean()
    df_features['irradiance_rolling_mean_3'] = df_features['Irradiance (W/m2)'].rolling(window=3, min_periods=1).mean()
    
    # Interaction features
    df_features['irradiance_temp'] = df_features['Irradiance (W/m2)'] * df_features['Temperature']
    df_features['irradiance_wind'] = df_features['Irradiance (W/m2)'] * df_features['Wind']
    
    return df_features

# Apply feature engineering
print("   Creating features for training data...")
train_enhanced = create_features(train_data)
print("   Creating features for testing data...")
test_enhanced = create_features(test_data)

# Remove rows with NaN values (due to lag features)
print("\n4. Cleaning data (removing NaN values from lag features)...")
train_clean = train_enhanced.dropna()
test_clean = test_enhanced.dropna()

print(f"Training data after cleaning: {train_clean.shape}")
print(f"Testing data after cleaning: {test_clean.shape}")

# Define feature columns for enhanced model
base_features = ['Irradiance (W/m2)', 'Temperature', 'Wind']
temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
lag_features = ['solar_lag1', 'irradiance_lag1', 'temperature_lag1', 'wind_lag1', 
                'solar_lag2', 'irradiance_lag2']
rolling_features = ['solar_rolling_mean_3', 'irradiance_rolling_mean_3']
interaction_features = ['irradiance_temp', 'irradiance_wind']

all_features = base_features + temporal_features + lag_features + rolling_features + interaction_features

print(f"\nFeature sets:")
print(f"  Base meteorological: {base_features}")
print(f"  Temporal encoding: {temporal_features}")
print(f"  Lag features: {lag_features}")
print(f"  Rolling features: {rolling_features}")
print(f"  Interaction features: {interaction_features}")
print(f"  Total features: {len(all_features)}")

# Prepare training and testing data
print("\n5. Preparing training and testing datasets...")
X_train = train_clean[all_features]
y_train = train_clean['Solar_Energy']
X_test = test_clean[all_features]
y_test = test_clean['Solar_Energy']

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Display feature statistics
print(f"\nEnhanced Training Data Statistics:")
print("Selected Features:")
print(X_train[all_features[:5]].describe())

# Train Enhanced Random Forest
print("\n6. Training Enhanced Random Forest Regressor...")
print(f"   Features: {len(all_features)} total features")
print("   Including: meteorological, temporal, lag, rolling, and interaction features")

enhanced_rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Fitting enhanced model...")
enhanced_rf.fit(X_train, y_train)
print("Enhanced model training completed!")

# Make predictions
print("\n7. Making predictions on test set...")
y_pred_enhanced = enhanced_rf.predict(X_test)

print(f"Predictions shape: {y_pred_enhanced.shape}")

# Calculate metrics
print("\n8. Calculating evaluation metrics...")

mae_enhanced = mean_absolute_error(y_test, y_pred_enhanced)
rmse_enhanced = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))
r2_enhanced = r2_score(y_test, y_pred_enhanced)

# Feature importance
feature_importance_enhanced = pd.DataFrame({
    'Feature': all_features,
    'Importance': enhanced_rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Display results
print("\n" + "=" * 60)
print("ENHANCED RANDOM FOREST RESULTS")
print("=" * 60)

print(f"\nEvaluation Metrics (Test Set):")
print(f"  MAE (Mean Absolute Error): {mae_enhanced:.4f} kWh")
print(f"  RMSE (Root Mean Squared Error): {rmse_enhanced:.4f} kWh")
print(f"  R² (R-squared): {r2_enhanced:.4f}")

print(f"\nInterpretation:")
print(f"  MAE: On average, predictions are off by {mae_enhanced:.4f} kWh")
print(f"  RMSE: Larger errors are penalized more heavily ({rmse_enhanced:.4f} kWh)")
if r2_enhanced > 0:
    print(f"  R²: Model explains {r2_enhanced*100:.1f}% of the variance in solar energy generation")
else:
    print(f"  R²: Model performs worse than simply predicting the mean ({r2_enhanced:.4f})")

# Feature Importance Analysis
print(f"\nTop 10 Most Important Features:")
print(feature_importance_enhanced.head(10))

print(f"\nFeature Importance by Category:")
categories = {
    'Base Meteorological': base_features,
    'Temporal Encoding': temporal_features,
    'Lag Features': lag_features,
    'Rolling Features': rolling_features,
    'Interaction Features': interaction_features
}

for category, features in categories.items():
    category_importance = feature_importance_enhanced[feature_importance_enhanced['Feature'].isin(features)]['Importance'].sum()
    print(f"  {category}: {category_importance:.3f}")

# Show sample predictions vs actual
print(f"\nSample Predictions vs Actual (first 10):")
comparison_df = pd.DataFrame({
    'Actual': y_test.head(10),
    'Predicted': y_pred_enhanced[:10],
    'Error': (y_test.head(10) - y_pred_enhanced[:10])
})
print(comparison_df)

# Compare with previous models
print(f"\nModel Comparison:")
print(f"  Enhanced Random Forest MAE: {mae_enhanced:.4f} kWh")
print(f"  Enhanced Random Forest RMSE: {rmse_enhanced:.4f} kWh")
print(f"  Enhanced Random Forest R²: {r2_enhanced:.4f}")

# Compare with basic Random Forest
try:
    basic_mae = 2.2211
    basic_rmse = 3.8949
    basic_r2 = 0.6242
    
    mae_improvement_basic = (basic_mae - mae_enhanced) / basic_mae * 100
    rmse_improvement_basic = (basic_rmse - rmse_enhanced) / basic_rmse * 100
    r2_improvement_basic = (r2_enhanced - basic_r2) / basic_r2 * 100
    
    print(f"\nImprovement vs Basic Random Forest:")
    print(f"  MAE Improvement: {mae_improvement_basic:+.1f}%")
    print(f"  RMSE Improvement: {rmse_improvement_basic:+.1f}%")
    print(f"  R² Improvement: {r2_improvement_basic:+.1f}%")
    
except:
    print(f"  (Basic Random Forest comparison not available)")

# Compare with persistence baseline
try:
    persistence_mae = 1.4402
    persistence_rmse = 2.6638
    persistence_r2 = 0.8242
    
    mae_improvement_persist = (persistence_mae - mae_enhanced) / persistence_mae * 100
    rmse_improvement_persist = (persistence_rmse - rmse_enhanced) / persistence_rmse * 100
    r2_improvement_persist = (r2_enhanced - persistence_r2) / persistence_r2 * 100
    
    print(f"\nImprovement vs Persistence Model:")
    print(f"  MAE Improvement: {mae_improvement_persist:+.1f}%")
    print(f"  RMSE Improvement: {rmse_improvement_persist:+.1f}%")
    print(f"  R² Improvement: {r2_improvement_persist:+.1f}%")
    
    if mae_improvement_persist > 0:
        print(f"  ✓ Enhanced RF improves MAE by {mae_improvement_persist:.1f}%")
    else:
        print(f"  ✗ Enhanced RF performs worse on MAE by {abs(mae_improvement_persist):.1f}%")
        
    if rmse_improvement_persist > 0:
        print(f"  ✓ Enhanced RF improves RMSE by {rmse_improvement_persist:.1f}%")
    else:
        print(f"  ✗ Enhanced RF performs worse on RMSE by {abs(rmse_improvement_persist):.1f}%")
        
    if r2_improvement_persist > 0:
        print(f"  ✓ Enhanced RF improves R² by {r2_improvement_persist:.1f}%")
    else:
        print(f"  ✗ Enhanced RF performs worse on R² by {abs(r2_improvement_persist):.1f}%")
        
except:
    print(f"  (Persistence baseline comparison not available)")

print("\n" + "=" * 60)
print("ENHANCED RANDOM FOREST EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nKey Improvements:")
print(f"  - Added {len(all_features)} total features (vs 3 base features)")
print(f"  - Included temporal encoding for day/night and seasonal patterns")
print(f"  - Added lag features to capture temporal dependencies")
print(f"  - Incorporated rolling averages and feature interactions")
print(f"\nBest performing feature categories:")
top_categories = feature_importance_enhanced.groupby(feature_importance_enhanced['Feature'].map(
    lambda x: next((cat for cat, feats in categories.items() if x in feats), 'Other')
))['Importance'].sum().sort_values(ascending=False)
for i, (category, importance) in enumerate(top_categories.head(3).items()):
    print(f"  {i+1}. {category}: {importance:.3f}")
print(f"\nRecommendation: Consider LSTM for better sequential modeling")