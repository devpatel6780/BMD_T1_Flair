# CNN Features + ML Approach: Usage Guide

## Overview

This approach extracts features using pretrained CNN, then trains traditional ML models (XGBoost, Random Forest, etc.) on those features. **Expected to outperform CNN end-to-end training on small datasets (192 samples).**

---

## Quick Start

### Step 1: Install Dependencies

```bash
# Install XGBoost (recommended - usually best performer)
pip install xgboost

# scikit-learn should already be installed
pip install scikit-learn
```

### Step 2: Extract Features (One Time)

```bash
python src/extract_features.py
```

**What it does:**
- Uses pretrained ResNet18 (no training, just feature extraction)
- Processes all images → extracts 512-dimensional feature vectors
- Saves to `features/` folder
- Takes ~5-10 minutes (one time)

**Output:**
- `features/features_train.npy` - Training features
- `features/features_val.npy` - Validation features
- `features/features_test.npy` - Test features
- `features/labels_*.npy` - Corresponding BMD labels
- `features/patient_ids_*.npy` - Patient IDs

---

### Step 3: Train ML Models

```bash
python src/train_ml_models.py
```

**What it does:**
- Loads extracted features
- Trains multiple models:
  - **XGBoost** (if installed - usually best)
  - **Random Forest**
  - **SVR** (Support Vector Regression)
  - **ElasticNet** (linear baseline)
- Evaluates on test set
- Saves models and results
- Takes ~2-5 minutes

**Output:**
- `models/xgboost_model.pkl` - Trained models
- `models/random_forest_model.pkl`
- `models/svr_model.pkl`
- `results_ml/ml_models_comparison.csv` - Comparison of all models
- `results_ml/<best_model>/` - Detailed results for best model

---

### Step 4: Compare with CNN (Optional)

```bash
python src/compare_ml_cnn.py
```

**What it does:**
- Compares ML results with CNN results
- Shows which approach is better
- Generates comparison report

**Output:**
- `results_ml/cnn_vs_ml_comparison.csv` - Side-by-side comparison

---

## Expected Results

| Approach | Expected Test R² | Training Time |
|----------|------------------|---------------|
| **CNN (end-to-end)** | 0.199 (current) | Hours |
| **XGBoost (CNN features)** | **0.35-0.55** ⭐ | Minutes |
| **Random Forest** | 0.30-0.50 | Minutes |
| **Ensemble** | **0.40-0.60** | Minutes |

---

## File Structure

After running all steps:

```
BMD_T1_Flair-main/
├── features/                      # NEW: Extracted features
│   ├── features_train.npy
│   ├── features_val.npy
│   ├── features_test.npy
│   ├── labels_*.npy
│   ├── patient_ids_*.npy
│   └── split_info.npy
│
├── models/                        # NEW: Trained ML models
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── svr_model.pkl
│   └── elasticnet_model.pkl
│
├── results_ml/                    # NEW: ML results
│   ├── ml_models_comparison.csv
│   ├── cnn_vs_ml_comparison.csv
│   ├── xgboost/                   # Best model results
│   │   ├── test_predictions.csv
│   │   ├── test_metrics.csv
│   │   └── *.png (plots)
│   └── ...
│
└── src/
    ├── extract_features.py        # NEW: Feature extraction
    ├── train_ml_models.py          # NEW: ML training
    └── compare_ml_cnn.py            # NEW: Comparison
```

---

## Detailed Workflow

### Feature Extraction Details

**Script:** `src/extract_features.py`

**Process:**
1. Loads pretrained ResNet18 (ImageNet weights)
2. Removes final classification layer
3. Freezes all layers (no training)
4. For each image:
   - Loads and preprocesses (same as CNN: crop, resize, normalize)
   - Passes through ResNet → 512-dim feature vector
   - Aggregates features per patient (average across scans)
5. Saves features to numpy files

**Key points:**
- ✅ Same preprocessing as CNN (fair comparison)
- ✅ Same patient-level split (no data leakage)
- ✅ No training needed (just forward pass)
- ✅ Fast (~5-10 minutes for all images)

---

### ML Training Details

**Script:** `src/train_ml_models.py`

**Models trained:**

1. **XGBoost** (if installed)
   - Usually best performer
   - Gradient boosting
   - Hyperparameters: n_estimators=200, max_depth=5, learning_rate=0.1

2. **Random Forest**
   - Robust, interpretable
   - Hyperparameters: n_estimators=200, max_depth=10

3. **SVR** (Support Vector Regression)
   - Good baseline
   - Uses RBF kernel
   - Features scaled for SVR

4. **ElasticNet**
   - Linear baseline
   - L1 + L2 regularization

**Process:**
1. Loads features and labels
2. Same train/val/test split as CNN
3. Trains all models
4. Evaluates on validation set → finds best model
5. Evaluates on test set
6. Saves models and results

**Key points:**
- ✅ Same evaluation metrics as CNN (R², MAE, RMSE, Pearson)
- ✅ Same data split (fair comparison)
- ✅ Multiple models for comparison
- ✅ Fast training (minutes)

---

### Comparison Details

**Script:** `src/compare_ml_cnn.py`

**Process:**
1. Finds CNN results (looks in common directories)
2. Loads ML results
3. Compares side-by-side:
   - Test R²
   - MAE, RMSE
   - Pearson correlation
4. Identifies best model
5. Shows improvement over CNN

**Key points:**
- ✅ Automatic comparison
- ✅ Shows best model
- ✅ Provides recommendations

---

## Troubleshooting

### Issue: XGBoost not installed

**Solution:**
```bash
pip install xgboost
```

**Note:** Other models (Random Forest, SVR) will still work.

---

### Issue: Features directory not found

**Error:** `FileNotFoundError: Features directory not found!`

**Solution:** Run feature extraction first:
```bash
python src/extract_features.py
```

---

### Issue: CNN results not found for comparison

**Error:** `Could not find CNN results`

**Solution:**
- Train CNN first, OR
- Specify CNN results directory in `compare_ml_cnn.py`

The script looks for results in:
- `full_finetune/`
- `regularized_v1/`
- `Modelresults1/`
- `results/`
- Any directory with `test_metrics.csv`

---

## Advanced Usage

### Hyperparameter Tuning

Edit `src/train_ml_models.py` to adjust hyperparameters:

```python
# XGBoost
model = xgb.XGBRegressor(
    n_estimators=300,      # More trees
    max_depth=7,           # Deeper trees
    learning_rate=0.05,    # Lower learning rate
    ...
)

# Random Forest
model = RandomForestRegressor(
    n_estimators=300,      # More trees
    max_depth=15,          # Deeper trees
    ...
)
```

### Using Different Feature Layers

Edit `src/extract_features.py` to extract from different layers:

```python
# Current: Extract from before final FC (512-dim)
# Alternative: Extract from after avgpool
# Or: Extract from multiple layers and concatenate
```

---

## Expected Timeline

| Step | Time | Frequency |
|------|------|-----------|
| Extract features | 5-10 min | Once |
| Train ML models | 2-5 min | Every run |
| Compare | <1 min | Once |

**Total first time:** ~10-15 minutes  
**Subsequent runs:** ~2-5 minutes (if features already extracted)

---

## Next Steps After Results

### If ML is Better (Expected):

1. **Use ML model** for final predictions
2. **Tune hyperparameters** to improve further
3. **Try ensemble** (combine multiple ML models)
4. **Add hand-crafted features** (texture, shape, etc.)

### If CNN is Better (Unlikely):

1. Review feature extraction
2. Try different feature layers
3. Consider ensemble (CNN + ML)

---

## Summary

**Quick workflow:**
```bash
# 1. Extract features (one time)
python src/extract_features.py

# 2. Train ML models
python src/train_ml_models.py

# 3. Compare with CNN
python src/compare_ml_cnn.py
```

**Expected outcome:**
- ML models should outperform CNN (R² 0.35-0.55 vs 0.199)
- Much faster training
- Better suited for small datasets

**Ready to try?** Start with Step 1!

