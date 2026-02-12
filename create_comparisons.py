import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("SOLAR PV FORECASTING - MODEL COMPARISON & VISUALIZATION")
print("=" * 60)

# Load test data
print("\nLoading test data...")
test_data = pd.read_csv('data/test_data_full.csv')
test_data['Time'] = pd.to_datetime(test_data['Time'])
test_data = test_data.set_index('Time')

print(f"Test data shape: {test_data.shape}")
print(f"Date range: {test_data.index.min()} to {test_data.index.max()}")

# Prepare features and target
feature_columns = ['Irradiance (W/m2)', 'Temperature', 'Wind', 'Wind Direction (degree)']
target_column = 'Solar_Energy'

# Get actual values
y_actual = test_data[target_column].values[24:]  # Skip first 24 hours for sequence alignment

# Function to load model predictions
def load_model_results(model_name, prediction_file=None):
    """Load or generate predictions for a specific model"""
    print(f"\nLoading {model_name} results...")
    
    if model_name == "Persistence":
        # Persistence model: predict previous hour's value
        y_pred = test_data[target_column].shift(1).fillna(0).values[1:]
        y_pred = y_pred[23:]  # Align with sequence models
        return y_pred
        
    elif model_name == "Basic Random Forest":
        # Run basic RF model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Load training data
        train_data = pd.read_csv('data/train_data_full.csv')
        train_data['Time'] = pd.to_datetime(train_data['Time'])
        train_data = train_data.set_index('Time')
        
        # Prepare data
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns][24:]  # Skip first 24 for alignment
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predict
        y_pred = rf_model.predict(X_test)
        return y_pred
        
    elif model_name == "Enhanced Random Forest":
        # Run enhanced RF model
        from sklearn.ensemble import RandomForestRegressor
        
        # Load training data
        train_data = pd.read_csv('data/train_data_full.csv')
        train_data['Time'] = pd.to_datetime(train_data['Time'])
        train_data = train_data.set_index('Time')
        
        def create_features(df):
            df_features = df.copy()
            # Temporal features
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features.index.hour / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features.index.hour / 24)
            df_features['day_of_year'] = df_features.index.dayofyear
            df_features['month'] = df_features.index.month
            # Lag features
            df_features['solar_lag1'] = df_features['Solar_Energy'].shift(1)
            df_features['solar_lag24'] = df_features['Solar_Energy'].shift(24)
            # Rolling features
            df_features['solar_rolling_mean_3'] = df_features['Solar_Energy'].rolling(window=3).mean()
            df_features['solar_rolling_mean_24'] = df_features['Solar_Energy'].rolling(window=24).mean()
            # Feature interactions
            df_features['irradiance_temp'] = df_features['Irradiance (W/m2)'] * df_features['Temperature']
            df_features['wind_irradiance'] = df_features['Wind'] * df_features['Irradiance (W/m2)']
            return df_features
        
        # Create features
        train_features = create_features(train_data)
        test_features = create_features(test_data)
        
        # Remove NaN values
        train_features = train_features.dropna()
        feature_cols = [col for col in train_features.columns if col != 'Solar_Energy']
        
        # Align test data
        test_features = test_features[feature_cols].dropna()
        y_test_aligned = test_data[target_column].iloc[test_features.index]
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        rf_model.fit(train_features[feature_cols], train_features['Solar_Energy'])
        
        # Predict
        y_pred = rf_model.predict(test_features)
        return y_pred, y_test_aligned.values
        
    elif model_name == "LSTM":
        # Run LSTM model (simplified version)
        print("Running LSTM model...")
        
        # Load training data
        train_data = pd.read_csv('data/train_data_full.csv')
        train_data['Time'] = pd.to_datetime(train_data['Time'])
        train_data = train_data.set_index('Time')
        
        def create_sequences(data, sequence_length=24):
            X, y = [], []
            all_features = feature_columns + [target_column]
            for i in range(sequence_length, len(data)):
                X.append(data[all_features].iloc[i-sequence_length:i].values)
                y.append(data[target_column].iloc[i])
            return np.array(X), np.array(y)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(train_data, sequence_length=24)
        X_test_seq, y_test_seq = create_sequences(test_data, sequence_length=24)
        
        # Normalize
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        train_features_flat = X_train_seq.reshape(-1, X_train_seq.shape[2])
        feature_scaler.fit(train_features_flat)
        
        X_train_scaled = np.array([feature_scaler.transform(seq) for seq in X_train_seq])
        X_test_scaled = np.array([feature_scaler.transform(seq) for seq in X_test_seq])
        y_train_scaled = target_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
        
        # Build model
        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(24, 5)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train (reduced epochs for speed)
        model.fit(X_train_scaled, y_train_scaled, epochs=15, batch_size=32, verbose=0)
        
        # Predict
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        return y_pred, y_test_seq

# Run all models and collect results
print("\n" + "="*60)
print("RUNNING ALL MODELS")
print("="*60)

model_results = {}
model_metrics = {}

# Run each model
models_to_run = [
    "Persistence",
    "Basic Random Forest", 
    "Enhanced Random Forest",
    "LSTM"
]

for model_name in models_to_run:
    try:
        if model_name in ["Enhanced Random Forest", "LSTM"]:
            y_pred, y_true = load_model_results(model_name)
            model_results[model_name] = {'y_pred': y_pred, 'y_true': y_true}
        else:
            y_pred = load_model_results(model_name)
            # For persistence and basic RF, align with test data
            if model_name == "Persistence":
                y_true = test_data[target_column].values[24:]
            else:  # Basic RF
                y_true = test_data[target_column].values[24:]
            model_results[model_name] = {'y_pred': y_pred, 'y_true': y_true}
            
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        model_metrics[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"✓ {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
    except Exception as e:
        print(f"✗ {model_name}: Error - {str(e)}")

# Create comprehensive visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Set up figure
fig = plt.figure(figsize=(20, 15))

# 1. Performance Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
models = list(model_metrics.keys())
mae_values = [model_metrics[model]['MAE'] for model in models]
rmse_values = [model_metrics[model]['RMSE'] for model in models]
r2_values = [model_metrics[model]['R2'] for model in models]

x = np.arange(len(models))
width = 0.25

ax1.bar(x - width, mae_values, width, label='MAE', alpha=0.8)
ax1.bar(x, rmse_values, width, label='RMSE', alpha=0.8)
ax1.bar(x + width, r2_values, width, label='R²', alpha=0.8)

ax1.set_xlabel('Models')
ax1.set_ylabel('Metric Values')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. MAE Comparison
ax2 = plt.subplot(2, 3, 2)
colors = ['red', 'blue', 'green', 'orange']
bars = ax2.bar(models, mae_values, color=colors, alpha=0.7)
ax2.set_ylabel('MAE (kWh)')
ax2.set_title('MAE Comparison')
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, mae_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 3. Time Series Comparison (sample)
ax3 = plt.subplot(2, 3, 3)
sample_size = 200  # Show first 200 points for clarity
sample_indices = test_data.index[24:24+sample_size]

for model_name in models_to_run:
    if model_name in model_results:
        y_pred_sample = model_results[model_name]['y_pred'][:sample_size]
        ax3.plot(sample_indices, y_pred_sample, label=f'{model_name}', alpha=0.7)

# Plot actual values
actual_sample = test_data[target_column].values[24:24+sample_size]
ax3.plot(sample_indices, actual_sample, 'black', linewidth=2, label='Actual', alpha=0.9)

ax3.set_xlabel('Time')
ax3.set_ylabel('Solar Energy (kWh)')
ax3.set_title('Time Series Prediction Comparison (Sample)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Scatter Plot - Actual vs Predicted (Best Model)
ax4 = plt.subplot(2, 3, 4)
best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['MAE'])
y_pred_best = model_results[best_model]['y_pred']
y_true_best = model_results[best_model]['y_true']

ax4.scatter(y_true_best, y_pred_best, alpha=0.5, s=1)
ax4.plot([y_true_best.min(), y_true_best.max()], [y_true_best.min(), y_true_best.max()], 'r--', linewidth=2)
ax4.set_xlabel('Actual Values (kWh)')
ax4.set_ylabel('Predicted Values (kWh)')
ax4.set_title(f'Actual vs Predicted - {best_model}')
ax4.grid(True, alpha=0.3)

# 5. Error Distribution
ax5 = plt.subplot(2, 3, 5)
for model_name in models_to_run:
    if model_name in model_results:
        errors = model_results[model_name]['y_true'] - model_results[model_name]['y_pred']
        ax5.hist(errors, bins=50, alpha=0.6, label=model_name, density=True)

ax5.set_xlabel('Prediction Error (kWh)')
ax5.set_ylabel('Density')
ax5.set_title('Error Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Performance Improvement Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create comparison table
table_data = []
for model in models:
    mae = model_metrics[model]['MAE']
    rmse = model_metrics[model]['RMSE']
    r2 = model_metrics[model]['R2']
    
    # Calculate improvement vs Persistence (baseline)
    baseline_mae = model_metrics['Persistence']['MAE']
    improvement_mae = ((baseline_mae - mae) / baseline_mae) * 100
    
    table_data.append([model, f'{mae:.4f}', f'{rmse:.4f}', f'{r2:.4f}', f'{improvement_mae:+.1f}%'])

table = ax6.table(cellText=table_data,
                  colLabels=['Model', 'MAE', 'RMSE', 'R²', 'MAE Improvement'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.15, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

ax6.set_title('Detailed Performance Comparison', pad=20)

plt.tight_layout()
plt.savefig('solar_forecasting_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'solar_forecasting_comparison.png'")

# Print summary
print("\n" + "="*60)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*60)

print("\nPerformance Metrics:")
print("-" * 50)
for model in models:
    metrics = model_metrics[model]
    print(f"{model:20} MAE: {metrics['MAE']:.4f}  RMSE: {metrics['RMSE']:.4f}  R²: {metrics['R2']:.4f}")

print("\nBest Performing Models:")
print("-" * 30)
best_mae = min(model_metrics.keys(), key=lambda x: model_metrics[x]['MAE'])
best_rmse = min(model_metrics.keys(), key=lambda x: model_metrics[x]['RMSE'])
best_r2 = max(model_metrics.keys(), key=lambda x: model_metrics[x]['R2'])

print(f"Lowest MAE: {best_mae} ({model_metrics[best_mae]['MAE']:.4f} kWh)")
print(f"Lowest RMSE: {best_rmse} ({model_metrics[best_rmse]['RMSE']:.4f} kWh)")
print(f"Highest R²: {best_r2} ({model_metrics[best_r2]['R2']:.4f})")

print(f"\nRecommendation: {best_mae} shows the best overall performance")

plt.show()
print("\n✓ All visualizations displayed successfully!")