import json

# Read the existing notebook
with open('model_development.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create new markdown cell for section header
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 13. Evaluate LSTM Model on Test Set"
    ]
}

# Create new code cell for LSTM evaluation
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generate predictions on test set\n",
        "y_pred_lstm_scaled = lstm_model.predict(X_test_lstm_scaled)\n",
        "y_pred_lstm = y_pred_lstm_scaled.flatten()  # Flatten to 1D array\n",
        "\n",
        "# Note: We don't need to inverse transform since we didn't scale the target variable\n",
        "# Only the input features were scaled\n",
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
        "plt.show()\n",
        "\n",
        "# Plot residuals\n",
        "residuals_lstm = y_test_lstm - y_pred_lstm\n",
        "plt.figure(figsize=(14, 5))\n",
        "plt.plot(residuals_lstm, linewidth=1.5, alpha=0.7, color='red')\n",
        "plt.axhline(y=0, color='black', linestyle='--', linewidth=1)\n",
        "plt.xlabel('Time Step', fontsize=12)\n",
        "plt.ylabel('Residual (kW)', fontsize=12)\n",
        "plt.title('LSTM Model Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Add the new cells to the notebook
notebook['cells'].append(markdown_cell)
notebook['cells'].append(code_cell)

# Save the updated notebook
with open('model_development.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully added LSTM evaluation section to the notebook!")
print(f"Total cells now: {len(notebook['cells'])}")
