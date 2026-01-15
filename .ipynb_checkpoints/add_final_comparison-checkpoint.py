import json

# Read the existing notebook
with open('model_development.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create new markdown cell for section header
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 14. Final Model Comparison"
    ]
}

# Create new code cell for final comparison table
code_cell = {
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

# Add the new cells to the notebook
notebook['cells'].append(markdown_cell)
notebook['cells'].append(code_cell)

# Save the updated notebook
with open('model_development.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully added final model comparison section to the notebook!")
print(f"Total cells now: {len(notebook['cells'])}")
