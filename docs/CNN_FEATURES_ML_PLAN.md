# CNN Features + Traditional ML: Implementation Plan

## Overview: What We'll Build

**Strategy:** Extract features using pretrained CNN → Train traditional ML models on features

**Why this works better:**
- Small dataset (192 samples) → Traditional ML > Deep Learning
- Pretrained CNN provides good features (no training needed)
- XGBoost works great on extracted features
- Expected: R² 0.35-0.55 (vs current 0.199)

---

## Complete Workflow

### Phase 1: Feature Extraction (Pretrained CNN)

```
Input: MRI images (.nii.gz files)
  ↓
Load images (same preprocessing as current: crop, resize, normalize)
  ↓
Pass through PRETRAINED ResNet18 (FREEZED, no training)
  ↓
Extract features from last layer (512-dimensional vectors)
  ↓
Save features to files (one per image)
```

**What happens:**
- Use pretrained ResNet18 (ImageNet weights)
- Freeze ALL layers (no training, just feature extraction)
- Forward pass through CNN → get feature vectors
- Save features for each image

**Output:**
- Feature matrix: [N_samples, 512] (or [N_samples, 1024])
- One file per patient, or one CSV with all features

---

### Phase 2: Train Traditional ML Models

```
Input: Feature vectors + BMD labels
  ↓
Split data (same patient-level split as CNN)
  ↓
Train multiple ML models:
  - XGBoost (usually best)
  - Random Forest
  - Support Vector Regression (SVR)
  - ElasticNet (linear baseline)
  ↓
Hyperparameter tuning (optional but recommended)
  ↓
Evaluate on test set
  ↓
Compare with CNN results
```

**What happens:**
- Load extracted features
- Load BMD labels (from metadata.xlsx)
- Same train/val/test split (patient-level, no data leakage)
- Train XGBoost, RF, SVR on features
- Predict BMD from features
- Evaluate same metrics (R², MAE, RMSE, Pearson)

**Output:**
- Trained models (pickle files)
- Predictions (CSV)
- Metrics (same format as CNN)
- Comparison with CNN results

---

### Phase 3: Ensemble (Optional, Best Results)

```
Input: Multiple model predictions
  ↓
Weighted average predictions:
  - CNN (end-to-end) prediction
  - XGBoost on CNN features prediction
  - RF on CNN features prediction
  ↓
Optimize weights on validation set
  ↓
Final ensemble prediction
```

**What happens:**
- Combine predictions from different models
- Weight by validation performance
- Final ensemble prediction

**Output:**
- Best possible R² (often 0.45-0.65)

---

## Detailed Step-by-Step Process

### Step 1: Feature Extraction Pipeline

**File: `src/extract_features.py`**

**Input:**
- Image files: `data/Sagittal_T1_FLAIR/*.nii.gz`
- Preprocessing: Same as current (center crop, resize, normalize)

**Process:**
1. Load pretrained ResNet18 (ImageNet weights)
2. Remove final classification layer
3. Freeze all layers (no training)
4. For each image:
   - Load and preprocess (same as DataLoader)
   - Pass through ResNet → get feature vector
   - Save feature vector

**Output:**
- `features/features_train.npy` - Training features [N_train, 512]
- `features/features_val.npy` - Validation features [N_val, 512]
- `features/features_test.npy` - Test features [N_test, 512]
- `features/patient_ids_train.npy` - Patient IDs for alignment

**Key Points:**
- ✅ No training needed (just forward pass)
- ✅ Same preprocessing as current CNN
- ✅ Patient-level split maintained
- ✅ Fast (minutes, not hours)

---

### Step 2: Traditional ML Training

**File: `src/train_ml_models.py`**

**Input:**
- Feature files: `features/*.npy`
- Labels: `data/metadata.xlsx` (BMD values)
- Same patient-level split as CNN

**Process:**
1. Load features and labels
2. Align features with patient IDs
3. Handle multi-scan patients (average features per patient)
4. Train models:
   ```python
   # XGBoost (usually best)
   xgb_model = XGBRegressor(...)
   xgb_model.fit(X_train, y_train)
   
   # Random Forest
   rf_model = RandomForestRegressor(...)
   rf_model.fit(X_train, y_train)
   
   # SVR
   svr_model = SVR(...)
   svr_model.fit(X_train, y_train)
   ```
5. Evaluate on test set
6. Save models and predictions

**Output:**
- Trained models: `models/xgboost_model.pkl`, `models/rf_model.pkl`, etc.
- Predictions: `results/ml_predictions.csv`
- Metrics: Same format as CNN (R², MAE, RMSE, Pearson)
- Comparison report: `results/ml_vs_cnn_comparison.txt`

**Key Points:**
- ✅ Same evaluation metrics as CNN (fair comparison)
- ✅ Same data split (fair comparison)
- ✅ Multiple models (XGBoost usually best)
- ✅ Fast training (minutes)

---

### Step 3: Evaluation and Comparison

**File: `src/compare_ml_cnn.py` (optional)**

**Process:**
1. Load CNN predictions (from previous training)
2. Load ML predictions (from Step 2)
3. Compare side-by-side:
   - Test R²
   - MAE, RMSE
   - Pearson correlation
   - Prediction examples
4. Generate comparison plots

**Output:**
- Comparison table (CNN vs ML)
- Comparison plots
- Recommendations

---

## File Structure

```
BMD_T1_Flair-main/
├── src/
│   ├── extract_features.py          # NEW: Extract CNN features
│   ├── train_ml_models.py           # NEW: Train XGBoost, RF, etc.
│   ├── compare_ml_cnn.py            # NEW: Compare results
│   ├── config.py                    # Existing
│   ├── DataLoader.py                # Existing (reuse preprocessing)
│   ├── Model_Transfer.py            # Existing
│   └── visualize_results.py         # Existing
│
├── features/                        # NEW: Extracted features
│   ├── features_train.npy
│   ├── features_val.npy
│   ├── features_test.npy
│   ├── patient_ids_train.npy
│   ├── patient_ids_val.npy
│   └── patient_ids_test.npy
│
├── models/                          # NEW: Trained ML models
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── svr_model.pkl
│
├── results/                         # Existing + NEW
│   ├── ml_predictions.csv           # NEW: ML predictions
│   ├── ml_metrics.csv               # NEW: ML metrics
│   └── comparison_report.txt        # NEW: CNN vs ML comparison
│
└── docs/
    └── CNN_FEATURES_ML_PLAN.md      # This file
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Feature Extraction                                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  MRI Images (.nii.gz)                                        │
│         │                                                     │
│         ▼                                                     │
│  Preprocessing (same as CNN)                                 │
│  - Load NIfTI                                                 │
│  - Extract 3 slices                                           │
│  - Center crop, resize, normalize                            │
│         │                                                     │
│         ▼                                                     │
│  Pretrained ResNet18 (FROZEN)                                │
│  - No training                                                │
│  - Just forward pass                                          │
│         │                                                     │
│         ▼                                                     │
│  Feature Vectors [512-dim]                                   │
│  - One per image                                              │
│  - Save to .npy files                                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: ML Training                                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Feature Vectors + BMD Labels                                │
│         │                                                     │
│         ▼                                                     │
│  Patient-level Split (same as CNN)                           │
│  - Train: 80%                                                 │
│  - Val: 10%                                                   │
│  - Test: 10%                                                  │
│         │                                                     │
│         ▼                                                     │
│  Aggregate per Patient                                        │
│  - Average features per patient                               │
│  - One prediction per patient                                 │
│         │                                                     │
│         ▼                                                     │
│  Train ML Models                                              │
│  - XGBoost (primary)                                          │
│  - Random Forest                                              │
│  - SVR                                                        │
│         │                                                     │
│         ▼                                                     │
│  Predictions + Metrics                                        │
│  - Same format as CNN                                         │
│  - R², MAE, RMSE, Pearson                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Comparison                                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  CNN Results + ML Results                                     │
│         │                                                     │
│         ▼                                                     │
│  Side-by-side Comparison                                      │
│  - Metrics comparison                                         │
│  - Prediction comparison                                      │
│  - Visualizations                                             │
│         │                                                     │
│         ▼                                                     │
│  Recommendation                                               │
│  - Which approach is better?                                  │
│  - Best model to use                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. **Reuse Existing Preprocessing**
- ✅ Use same DataLoader preprocessing
- ✅ Same center crop, resize, normalization
- ✅ Same slice extraction (N_SLICES=3)
- ✅ Fair comparison with CNN

### 2. **Same Data Split**
- ✅ Use same patient-level split
- ✅ Same train/val/test patients
- ✅ No data leakage
- ✅ Direct comparison possible

### 3. **Same Evaluation Metrics**
- ✅ R², MAE, RMSE, Pearson correlation
- ✅ Per-patient aggregation
- ✅ Same format as CNN results
- ✅ Easy comparison

### 4. **Patient-Level Handling**
- ✅ Multiple scans per patient → average features
- ✅ Same aggregation as CNN
- ✅ One prediction per patient

---

## Implementation Details

### Feature Extraction

**Which layer to extract from?**
- Option 1: Before final FC (512-dim for ResNet18) ⭐ Recommended
- Option 2: After avgpool (512-dim) ⭐ Also good
- Option 3: Multiple layers (concatenate) - More complex

**Recommendation:** Extract from layer just before final FC (features)

**Code structure:**
```python
# Load pretrained ResNet
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.eval()  # Set to eval mode
# Remove final layer
features_model = nn.Sequential(*list(resnet.children())[:-1])

# Freeze all
for param in features_model.parameters():
    param.requires_grad = False

# Extract features
with torch.no_grad():
    features = features_model(images)  # [B, 512, 1, 1]
    features = features.squeeze()      # [B, 512]
```

---

### ML Model Selection

**Primary Model: XGBoost**
- Usually best for regression
- Good with small datasets
- Fast training
- Hyperparameter tuning available

**Secondary Models:**
- Random Forest (robust, interpretable)
- SVR (good baseline)
- ElasticNet (linear baseline)

**Hyperparameter Tuning:**
- Grid search or random search
- Cross-validation on validation set
- Focus on XGBoost (most important)

---

### Handling Multi-Scan Patients

**Same as CNN:**
1. Extract features for all scans
2. Average features per patient
3. Train on averaged features
4. Predict per patient

**Code:**
```python
# Group by patient ID
patient_features = {}
for pid, features in zip(patient_ids, features_list):
    if pid not in patient_features:
        patient_features[pid] = []
    patient_features[pid].append(features)

# Average per patient
avg_features = {pid: np.mean(feats, axis=0) 
                for pid, feats in patient_features.items()}
```

---

## Usage Workflow

### Step 1: Extract Features (One Time)

```bash
python src/extract_features.py
```

**What it does:**
- Loads all images
- Extracts features using pretrained ResNet
- Saves to `features/` folder
- Takes ~5-10 minutes (one time)

---

### Step 2: Train ML Models

```bash
python src/train_ml_models.py
```

**What it does:**
- Loads extracted features
- Trains XGBoost, RF, SVR
- Evaluates on test set
- Saves models and results
- Takes ~2-5 minutes

---

### Step 3: Compare (Optional)

```bash
python src/compare_ml_cnn.py
```

**What it does:**
- Compares ML vs CNN results
- Generates comparison report
- Shows which is better

---

## Expected Results

### Comparison Table

| Model | Test R² | MAE | RMSE | Pearson | Training Time |
|-------|---------|-----|------|---------|---------------|
| **CNN (current)** | 0.199 | 0.125 | 0.166 | 0.478 | ~Hours |
| **XGBoost (expected)** | **0.35-0.55** | **0.10-0.12** | **0.13-0.16** | **0.60-0.75** | ~Minutes |
| **RF (expected)** | 0.30-0.50 | 0.11-0.13 | 0.14-0.17 | 0.55-0.70 | ~Minutes |
| **Ensemble (expected)** | **0.40-0.60** | **0.09-0.11** | **0.12-0.15** | **0.65-0.80** | ~Minutes |

---

## Dependencies to Add

```python
# In requirements.txt or install separately
scikit-learn>=1.0.0
xgboost>=1.5.0
numpy>=1.21.0
pandas>=1.3.0
```

**Install:**
```bash
pip install xgboost scikit-learn
```

---

## Advantages of This Approach

1. ✅ **Better results** (expected R² 0.35-0.55 vs 0.199)
2. ✅ **Faster** (minutes vs hours)
3. ✅ **More interpretable** (feature importance)
4. ✅ **Suitable for small datasets** (192 samples)
5. ✅ **Standard approach** for medical imaging with small data
6. ✅ **Easy to iterate** (fast training)

---

## Next Steps After Implementation

1. **Compare results** - See if ML beats CNN
2. **If ML is better:**
   - Use ML for final model
   - Tune hyperparameters
   - Add more features (if needed)
3. **If similar:**
   - Consider ensemble
   - Combine CNN + ML predictions
4. **Publication:**
   - Report both approaches
   - Show ML is better (likely)
   - Discuss why (small dataset)

---

## Summary

**What we'll build:**
1. Feature extraction script (pretrained CNN → features)
2. ML training script (XGBoost, RF, SVR on features)
3. Comparison script (CNN vs ML)

**How it works:**
1. Extract features (one time, ~5-10 min)
2. Train ML models (each run, ~2-5 min)
3. Compare with CNN (one time)

**Expected outcome:**
- ML approach likely beats CNN (R² 0.35-0.55 vs 0.199)
- Much faster to train
- Better suited for small datasets

**Ready to implement?** Let me know and I'll create all three scripts!

