import json

notebook = {
    "cells": [
        # Cell 0: Title
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Solar PV Forecasting - Model Development\n", "\n", "This notebook implements complete forecasting pipeline with Persistence, Random Forest, and LSTM models."]
        },
        
        # Cell 1: Imports Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Import Libraries"]
        },
        
        # Cell 2: Imports Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
                "from sklearn.preprocessing import MinMaxScaler\n",
                "from tensorflow.keras.models import Sequential\n",
                "from tensorflow.keras.layers import LSTM, Dense\n",
                "from tensorflow.keras.callbacks import EarlyStopping\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "print(\"Libraries imported successfully!\")"
            ]
        },
        
        # Cell 3: Load Data Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Load Cleaned Dataset"]
        },
        
        # Cell 4: Load Data Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the cleaned dataset with datetime index\n",
                "df = pd.read_csv('data/solar_pv_clean_hourly.csv', index_col=0, parse_dates=True)\n",
                "\n",
                "# Display dataset shape\n",
                "print(f\"Dataset Shape: {df.shape}\")\n",
                "\n",
                "# Display column names\n",
                "print(f\"\\nColumn Names:\")\n",
                "print(df.columns.tolist())\n",
                "\n",
                "# Display date range\n",
                "print(f\"\\nDate Range:\")\n",
                "print(f\"Start: {df.index.min()}\")\n",
                "print(f\"End: {df.index.max()}\")\n",
                "\n",
                "# Display first few rows\n",
                "df.head()"
            ]
        },
        
        # Cell 5: Train-Test Split Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Train-Test Split"]
        },
        
        # Cell 6: Train-Test Split Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate split point for 80-20 split\n",
                "split_idx = int(len(df) * 0.8)\n",
                "\n",
                "# Split data chronologically (no shuffle)\n",
                "train_df = df.iloc[:split_idx]\n",
                "test_df = df.iloc[split_idx:]\n",
                "\n",
                "# Separate features and target\n",
                "X_train = train_df.drop('DC_POWER', axis=1)\n",
                "y_train = train_df['DC_POWER']\n",
                "\n",
                "X_test = test_df.drop('DC_POWER', axis=1)\n",
                "y_test = test_df['DC_POWER']\n",
                "\n",
                "# Print shapes\n",
                "print(f\"X_train shape: {X_train.shape}\")\n",
                "print(f\"X_test shape: {X_test.shape}\")\n",
                "print(f\"y_train shape: {y_train.shape}\")\n",
                "print(f\"y_test shape: {y_test.shape}\")\n",
                "\n",
                "# Print date ranges for train and test sets\n",
                "print(f\"\\nTrain date range: {X_train.index.min()} to {X_train.index.max()}\")\n",
                "print(f\"Test date range: {X_test.index.min()} to {X_test.index.max()}\")"
            ]
        },
        
        # Cell 7: Persistence Baseline Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Persistence Baseline Model"]
        },
        
        # Cell 8: Persistence Baseline Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Persistence model: predict next hour = current hour\n",
                "y_pred_persistence = y_test.shift(1).dropna()\n",
                "y_test_persistence = y_test.iloc[1:]\n",
                "\n",
                "# Compute metrics\n",
                "mae_persistence = np.mean(np.abs(y_test_persistence - y_pred_persistence))\n",
                "rmse_persistence = np.sqrt(np.mean((y_test_persistence - y_pred_persistence)**2))\n",
                "mape_persistence = np.mean(np.abs((y_test_persistence - y_pred_persistence) / (y_test_persistence + 1e-8))) * 100\n",
                "\n",
                "print(\"\\n\" + \"=\"*50)\n",
                "print(\"Persistence Baseline Model - Test Set Performance\")\n",
                "print(\"=\"*50)\n",
                "print(f\"MAE:  {mae_persistence:.4f} kW\")\n",
                "print(f\"RMSE: {rmse_persistence:.4f} kW\")\n",
                "print(f\"MAPE: {mape_persistence:.2f}%\")\n",
                "print(\"=\"*50)"
            ]
        },
        
        # Cell 9: Feature Engineering Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Feature Engineering"]
        },
        
        # Cell 10: Feature Engineering Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create lag features\n",
                "df['DC_POWER_lag1'] = df['DC_POWER'].shift(1)\n",
                "df['DC_POWER_lag2'] = df['DC_POWER'].shift(2)\n",
                "\n",
                "# Create hour sine and cosine features\n",
                "df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)\n",
                "df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)\n",
                "\n",
                "# Drop rows with NaN values\n",
                "df_clean = df.dropna()\n",
                "\n",
                "# Update train-test split with engineered features\n",
                "split_idx = int(len(df_clean) * 0.8)\n",
                "train_df = df_clean.iloc[:split_idx]\n",
                "test_df = df_clean.iloc[split_idx:]\n",
                "\n",
                "X_train = train_df.drop('DC_POWER', axis=1)\n",
                "y_train = train_df['DC_POWER']\n",
                "X_test = test_df.drop('DC_POWER', axis=1)\n",
                "y_test = test_df['DC_POWER']\n",
                "\n",
                "print(f\"\\nFinal feature set: {list(X_train.columns)}\")\n",
                "print(f\"\\nX_train shape: {X_train.shape}\")\n",
                "print(f\"X_test shape: {X_test.shape}\")"
            ]
        },
        
        # Cell 11: Random Forest Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Random Forest Model"]
        },
        
        # Cell 12: Random Forest Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train Random Forest model\n",
                "rf_model = RandomForestRegressor(random_state=42)\n",
                "rf_model.fit(X_train, y_train)\n",
                "\n",
                "# Make predictions\n",
                "y_pred_rf = rf_model.predict(X_test)\n",
                "\n",
                "# Compute metrics\n",
                "mae_rf = mean_absolute_error(y_test, y_pred_rf)\n",
                "rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))\n",
                "mape_rf = np.mean(np.abs((y_test - y_pred_rf) / (y_test + 1e-8))) * 100\n",
                "\n",
                "print(\"\\n\" + \"=\"*50)\n",
                "print(\"Random Forest Model - Test Set Performance\")\n",
                "print(\"=\"*50)\n",
                "print(f\"MAE:  {mae_rf:.4f} kW\")\n",
                "print(f\"RMSE: {rmse_rf:.4f} kW\")\n",
                "print(f\"MAPE: {mape_rf:.2f}%\")\n",
                "print(\"=\"*50)\n",
                "\n",
                "# Plot actual vs predicted\n",
                "plt.figure(figsize=(14, 6))\n",
                "plt.plot(y_test.values, label='Actual DC_POWER', linewidth=2, alpha=0.7)\n",
                "plt.plot(y_pred_rf, label='Random Forest Predicted', linewidth=2, alpha=0.7)\n",
                "plt.xlabel('Time Step', fontsize=12)\n",
                "plt.ylabel('DC_POWER (kW)', fontsize=12)\n",
                "plt.title('Random Forest: Actual vs Predicted DC_POWER (Test Set)', fontsize=14, fontweight='bold')\n",
                "plt.legend(fontsize=11)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        
        # Cell 13: LSTM Prep Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. LSTM Data Preparation"]
        },
        
        # Cell 14: LSTM Prep Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_sequences(data, lookback=24):\n",
                "    X, y = [], []\n",
                "    for i in range(lookback, len(data)):\n",
                "        X.append(data[i-lookback:i, :])  # Past 24 hours of all features\n",
                "        y.append(data[i, 0])  # Next hour DC_POWER (first column)\n",
                "    return np.array(X), np.array(y)\n",
                "\n",
                "# Selected features for LSTM\n",
                "lstm_features = ['DC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE']\n",
                "lookback = 24\n",
                "\n",
                "# Prepare data\n",
                "train_data = train_df[lstm_features].values\n",
                "test_data = test_df[lstm_features].values\n",
                "\n",
                "# Create sequences\n",
                "X_train_lstm, y_train_lstm = create_sequences(train_data, lookback)\n",
                "X_test_lstm, y_test_lstm = create_sequences(test_data, lookback)\n",
                "\n",
                "print(f\"X_train_lstm shape: {X_train_lstm.shape}\")\n",
                "print(f\"y_train_lstm shape: {y_train_lstm.shape}\")\n",
                "print(f\"X_test_lstm shape: {X_test_lstm.shape}\")\n",
                "print(f\"y_test_lstm shape: {y_test_lstm.shape}\")"
            ]
        },
        
        # Cell 15: LSTM Scaling Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Scale LSTM Features"]
        },
        
        # Cell 16: LSTM Scaling Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize MinMaxScaler\n",
                "scaler = MinMaxScaler()\n",
                "\n",
                "# Fit scaler on training data ONLY\n",
                "n_samples_train, n_timesteps, n_features = X_train_lstm.shape\n",
                "X_train_reshaped = X_train_lstm.reshape(-1, n_features)\n",
                "scaler.fit(X_train_reshaped)\n",
                "\n",
                "# Transform training data\n",
                "X_train_scaled = scaler.transform(X_train_reshaped)\n",
                "X_train_lstm_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)\n",
                "\n",
                "# Transform test data using same scaler (NO FITTING)\n",
                "n_samples_test = X_test_lstm.shape[0]\n",
                "X_test_reshaped = X_test_lstm.reshape(-1, n_features)\n",
                "X_test_scaled = scaler.transform(X_test_reshaped)\n",
                "X_test_lstm_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)\n",
                "\n",
                "print(\"Feature scaling completed!\")\n",
                "print(f\"Scaled training data shape: {X_train_lstm_scaled.shape}\")\n",
                "print(f\"Scaled test data shape: {X_test_lstm_scaled.shape}\")"
            ]
        },
        
        # Cell 17: LSTM Build Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Build LSTM Model"]
        },
        
        # Cell 18: LSTM Build Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build LSTM model\n",
                "lstm_model = Sequential([\n",
                "    LSTM(50, input_shape=(lookback, len(lstm_features))),\n",
                "    Dense(1)\n",
                "])\n",
                "\n",
                "lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])\n",
                "\n",
                "print(\"\\nLSTM Model Summary:\")\n",
                "lstm_model.summary()"
            ]
        },
        
        # Cell 19: LSTM Train Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Train LSTM Model"]
        },
        
        # Cell 20: LSTM Train Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup early stopping\n",
                "early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)\n",
                "\n",
                "# Train LSTM model\n",
                "history = lstm_model.fit(\n",
                "    X_train_lstm_scaled,\n",
                "    y_train_lstm,\n",
                "    epochs=25,\n",
                "    batch_size=32,\n",
                "    validation_split=0.2,\n",
                "    callbacks=[early_stop],\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "# Plot training and validation loss\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.plot(history.history['loss'], label='Training Loss', linewidth=2)\n",
                "plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)\n",
                "plt.xlabel('Epoch', fontsize=12)\n",
                "plt.ylabel('Loss (MSE)', fontsize=12)\n",
                "plt.title('LSTM Model Training History', fontsize=14, fontweight='bold')\n",
                "plt.legend(fontsize=11)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        
        # Cell 21: LSTM Eval Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Evaluate LSTM Model"]
        },
        
        # Cell 22: LSTM Eval Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate predictions on test set\n",
                "y_pred_lstm_scaled = lstm_model.predict(X_test_lstm_scaled)\n",
                "y_pred_lstm = y_pred_lstm_scaled.flatten()  # Flatten to 1D array\n",
                "\n",
                "# Compute evaluation metrics\n",
                "mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)\n",
                "rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))\n",
                "mape_lstm = np.mean(np.abs((y_test_lstm - y_pred_lstm) / (y_test_lstm + 1e-8))) * 100\n",
                "\n",
                "# Print metrics\n",
                "print(\"\\n\" + \"=\"*50)\n",
                "print(\"LSTM Model - Test Set Performance\")\n",
                "print(\"=\"*50)\n",
                "print(f\"MAE:  {mae_lstm:.4f} kW\")\n",
                "print(f\"RMSE: {rmse_lstm:.4f} kW\")\n",
                "print(f\"MAPE: {mape_lstm:.2f}%\")\n",
                "print(\"=\"*50)\n",
                "\n",
                "# Plot actual vs predicted\n",
                "plt.figure(figsize=(14, 6))\n",
                "plt.plot(y_test_lstm, label='Actual DC_POWER', linewidth=2, alpha=0.7)\n",
                "plt.plot(y_pred_lstm, label='LSTM Predicted', linewidth=2, alpha=0.7)\n",
                "plt.xlabel('Time Step', fontsize=12)\n",
                "plt.ylabel('DC_POWER (kW)', fontsize=12)\n",
                "plt.title('LSTM Model: Actual vs Predicted DC_POWER (Test Set)', fontsize=14, fontweight='bold')\n",
                "plt.legend(fontsize=11)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        
        # Cell 23: Final Comparison Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 12. Final Model Comparison"]
        },
        
        # Cell 24: Final Comparison Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create comprehensive comparison table\n",
                "comparison_df = pd.DataFrame({\n",
                "    'Model': ['Persistence Baseline', 'Random Forest', 'LSTM'],\n",
                "    'MAE (kW)': [mae_persistence, mae_rf, mae_lstm],\n",
                "    'RMSE (kW)': [rmse_persistence, rmse_rf, rmse_lstm],\n",
                "    'MAPE (%)': [mape_persistence, mape_rf, mape_lstm]\n",
                "})\n",
                "\n",
                "# Display the comparison table\n",
                "print(\"\\n\" + \"=\"*70)\n",
                "print(\"FINAL MODEL COMPARISON - Test Set Performance\")\n",
                "print(\"=\"*70)\n",
                "print(comparison_df.to_string(index=False))\n",
                "print(\"=\"*70)\n",
                "\n",
                "# Find best model for each metric\n",
                "best_mae = comparison_df.loc[comparison_df['MAE (kW)'].idxmin(), 'Model']\n",
                "best_rmse = comparison_df.loc[comparison_df['RMSE (kW)'].idxmin(), 'Model']\n",
                "best_mape = comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model']\n",
                "\n",
                "print(f\"\\nBest Model by MAE:  {best_mae}\")\n",
                "print(f\"Best Model by RMSE: {best_rmse}\")\n",
                "print(f\"Best Model by MAPE: {best_mape}\")\n",
                "\n",
                "# Visualize comparison\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                "\n",
                "# MAE comparison\n",
                "axes[0].bar(comparison_df['Model'], comparison_df['MAE (kW)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])\n",
                "axes[0].set_ylabel('MAE (kW)', fontsize=11)\n",
                "axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')\n",
                "axes[0].tick_params(axis='x', rotation=15)\n",
                "axes[0].grid(axis='y', alpha=0.3)\n",
                "\n",
                "# RMSE comparison\n",
                "axes[1].bar(comparison_df['Model'], comparison_df['RMSE (kW)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])\n",
                "axes[1].set_ylabel('RMSE (kW)', fontsize=11)\n",
                "axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')\n",
                "axes[1].tick_params(axis='x', rotation=15)\n",
                "axes[1].grid(axis='y', alpha=0.3)\n",
                "\n",
                "# MAPE comparison\n",
                "axes[2].bar(comparison_df['Model'], comparison_df['MAPE (%)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])\n",
                "axes[2].set_ylabel('MAPE (%)', fontsize=11)\n",
                "axes[2].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')\n",
                "axes[2].tick_params(axis='x', rotation=15)\n",
                "axes[2].grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the complete notebook
with open('model_development.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Complete notebook created successfully!")
print(f"Total cells: {len(notebook['cells'])}")
print("\nSections included:")
print("1. Import Libraries")
print("2. Load Cleaned Dataset")
print("3. Train-Test Split")
print("4. Persistence Baseline Model")
print("5. Feature Engineering")
print("6. Random Forest Model")
print("7. LSTM Data Preparation")
print("8. Scale LSTM Features")
print("9. Build LSTM Model")
print("10. Train LSTM Model")
print("11. Evaluate LSTM Model")
print("12. Final Model Comparison")
