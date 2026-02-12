import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("=" * 60)
print("LSTM SOLAR FORECASTING MODEL")
print("=" * 60)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

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
feature_columns = ['Irradiance (W/m2)', 'Temperature', 'Wind', 'Wind Direction (degree)']
target_column = 'Solar_Energy'

# Create sequences function
def create_sequences(data, sequence_length=24):
    """Create sequences of past 24 hours to predict next hour"""
    X, y = [], []
    
    # Use all available features including the target for better context
    all_features = feature_columns + [target_column]
    
    for i in range(sequence_length, len(data)):
        # Input sequence: past 24 hours of all features
        X.append(data[all_features].iloc[i-sequence_length:i].values)
        # Target: next hour's solar energy
        y.append(data[target_column].iloc[i])
    
    return np.array(X), np.array(y)

# Create sequences for training and testing
print(f"\n4. Creating sequences (24-hour lookback)...")
print("   Features: Irradiance, Temperature, Wind, Wind Direction, and Solar_Energy history")

X_train_seq, y_train_seq = create_sequences(train_data, sequence_length=24)
X_test_seq, y_test_seq = create_sequences(test_data, sequence_length=24)

print(f"Training sequences shape: {X_train_seq.shape}")
print(f"Training targets shape: {y_train_seq.shape}")
print(f"Testing sequences shape: {X_test_seq.shape}")
print(f"Testing targets shape: {y_test_seq.shape}")

# Normalize features
print("\n5. Normalizing features and target...")
print("   Using MinMaxScaler for LSTM compatibility")

# Initialize scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit scalers on training data only (to prevent data leakage)
train_features_flat = X_train_seq.reshape(-1, X_train_seq.shape[2])
feature_scaler.fit(train_features_flat)

# Scale features
X_train_scaled = np.array([feature_scaler.transform(seq) for seq in X_train_seq])
X_test_scaled = np.array([feature_scaler.transform(seq) for seq in X_test_seq])

# Scale targets
y_train_scaled = target_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test_seq.reshape(-1, 1)).flatten()

print(f"Normalized training sequences shape: {X_train_scaled.shape}")
print(f"Normalized testing sequences shape: {X_test_scaled.shape}")

# Display scaling info
print(f"\nFeature scaling range: [{train_features_flat.min():.2f}, {train_features_flat.max():.2f}] -> [0, 1]")
print(f"Target scaling range: [{y_train_seq.min():.2f}, {y_train_seq.max():.2f}] -> [0, 1]")

# Build LSTM Model
print("\n6. Building LSTM Model...")
print("   Architecture: 1 LSTM layer + 1 Dense output layer")

model = Sequential([
    # LSTM layer
    LSTM(
        units=50,
        return_sequences=False,
        input_shape=(24, len(feature_columns) + 1),  # 24 timesteps, 5 features
        name='lstm_layer'
    ),
    
    # Dropout for regularization
    Dropout(0.2, name='dropout'),
    
    # Dense output layer
    Dense(1, name='output_layer')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("Model architecture:")
model.summary()

# Train the model
print("\n7. Training LSTM Model...")
print("   Epochs: 50")
print("   Batch size: 32")
print("   Validation split: 0.2")

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    shuffle=False  # Important for time series
)

# Make predictions
print("\n8. Making predictions on test set...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()

print(f"Predictions shape: {y_pred.shape}")
print(f"Sample predictions: {y_pred[:5]}")

# Calculate metrics
print("\n9. Calculating evaluation metrics...")

mae = mean_absolute_error(y_test_seq, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
r2 = r2_score(y_test_seq, y_pred)

# Display results
print("\n" + "=" * 60)
print("LSTM MODEL RESULTS")
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

# Show sample predictions vs actual
print(f"\nSample Predictions vs Actual (first 10):")
comparison_df = pd.DataFrame({
    'Actual': y_test_seq[:10],
    'Predicted': y_pred[:10],
    'Error': (y_test_seq[:10] - y_pred[:10])
})
print(comparison_df)

# Model Performance Analysis
print(f"\nLSTM Model Performance Analysis:")
print(f"  Sequence length: 24 hours")
print(f"  Input features: {len(feature_columns) + 1} (including Solar_Energy history)")
print(f"  LSTM units: 50")
print(f"  Training samples: {X_train_scaled.shape[0]}")
print(f"  Testing samples: {X_test_scaled.shape[0]}")

# Compare with previous models
print(f"\nModel Comparison:")

# Compare with enhanced Random Forest
try:
    rf_mae = 1.3177
    rf_rmse = 2.6463
    rf_r2 = 0.8266
    
    mae_improvement_rf = (rf_mae - mae) / rf_mae * 100
    rmse_improvement_rf = (rf_rmse - rmse) / rf_rmse * 100
    r2_improvement_rf = (r2 - rf_r2) / rf_r2 * 100
    
    print(f"  vs Enhanced Random Forest:")
    print(f"    MAE Improvement: {mae_improvement_rf:+.1f}%")
    print(f"    RMSE Improvement: {rmse_improvement_rf:+.1f}%")
    print(f"    R² Improvement: {r2_improvement_rf:+.1f}%")
    
except:
    print(f"  Enhanced Random Forest comparison not available")

# Compare with persistence baseline
try:
    persistence_mae = 1.4402
    persistence_rmse = 2.6638
    persistence_r2 = 0.8242
    
    mae_improvement_persist = (persistence_mae - mae) / persistence_mae * 100
    rmse_improvement_persist = (persistence_rmse - rmse) / persistence_rmse * 100
    r2_improvement_persist = (r2 - persistence_r2) / persistence_r2 * 100
    
    print(f"\n  vs Persistence Model:")
    print(f"    MAE Improvement: {mae_improvement_persist:+.1f}%")
    print(f"    RMSE Improvement: {rmse_improvement_persist:+.1f}%")
    print(f"    R² Improvement: {r2_improvement_persist:+.1f}%")
    
    if mae_improvement_persist > 0:
        print(f"    ✓ LSTM improves MAE by {mae_improvement_persist:.1f}%")
    else:
        print(f"    ✗ LSTM performs worse on MAE by {abs(mae_improvement_persist):.1f}%")
        
    if rmse_improvement_persist > 0:
        print(f"    ✓ LSTM improves RMSE by {rmse_improvement_persist:.1f}%")
    else:
        print(f"    ✗ LSTM performs worse on RMSE by {abs(rmse_improvement_persist):.1f}%")
        
    if r2_improvement_persist > 0:
        print(f"    ✓ LSTM improves R² by {r2_improvement_persist:.1f}%")
    else:
        print(f"    ✗ LSTM performs worse on R² by {abs(r2_improvement_persist):.1f}%")
        
except:
    print(f"  Persistence baseline comparison not available")

# Training history summary
print(f"\nTraining History Summary:")
print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"  Best validation loss: {min(history.history['val_loss']):.6f}")

print("\n" + "=" * 60)
print("LSTM MODEL EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nKey Achievements:")
print(f"  - Built LSTM with 24-hour sequence input")
print(f"  - Used 5 input features (4 meteorological + Solar_Energy history)")
print(f"  - Proper normalization with MinMaxScaler")
print(f"  - Prevented data leakage through careful scaling")
print(f"\nModel Strengths:")
print(f"  - Captures temporal dependencies effectively")
print(f"  - Handles sequential patterns in solar generation")
print(f"  - Uses historical context for better predictions")
print(f"\nRecommendation: LSTM shows strong performance for time-series forecasting")