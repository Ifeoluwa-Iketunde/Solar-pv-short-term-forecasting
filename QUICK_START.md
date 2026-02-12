# Quick Start Guide

## 🚀 Fast Track to Running the Project

### Option 1: Run Everything Automatically (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py
```

This will:
1. Process all raw data files
2. Create train/test splits
3. Run all 4 models
4. Display comparative results

### Option 2: Run Individual Steps

```bash
# 1. Process data
python scripts/create_final_full_dataset.py

# 2. Split data
python scripts/temporal_split_full.py

# 3. Run models (in any order)
python scripts/persistence_model.py
python scripts/random_forest_model.py
python scripts/enhanced_random_forest.py
python scripts/lstm_model.py
```

## 📋 What You'll Get

After running the pipeline, you'll have:
- **Processed datasets** in the `data/` folder
- **Model performance results** printed to console
- **Comparative analysis** of all approaches
- **Ready-to-use models** for forecasting

## 🎯 Expected Results

The pipeline will show you:
- **Persistence Model**: Baseline performance (~1.44 MAE)
- **Basic Random Forest**: Simple ML approach (~2.21 MAE)
- **Enhanced Random Forest**: Best performer (~1.32 MAE)
- **LSTM Model**: Deep learning approach (~1.63 MAE)

## 📚 Need Help?

Check the detailed README.md for:
- Complete project documentation
- Technical explanations
- Model architecture details
- Performance analysis

## ⚡ Quick Troubleshooting

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**Memory issues with LSTM?**
- Reduce batch size in `scripts/lstm_model.py`
- Use smaller sequence length (currently 24 hours)

**Slow execution?**
- The pipeline takes 10-15 minutes total
- LSTM training is the slowest part (~5-8 minutes)