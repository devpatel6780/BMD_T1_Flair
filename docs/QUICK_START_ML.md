# Quick Start: CNN Features + ML Approach

## What Was Created

✅ **3 new scripts:**
1. `src/extract_features.py` - Extract CNN features
2. `src/train_ml_models.py` - Train ML models
3. `src/compare_ml_cnn.py` - Compare results

---

## Quick Start (3 Steps)

### Step 1: Extract Features (One Time, ~5-10 min)

```bash
python src/extract_features.py
```

**What happens:**
- Uses pretrained ResNet18 (no training)
- Extracts 512-dim features from all images
- Saves to `features/` folder
- Same preprocessing as CNN (fair comparison)

---

### Step 2: Train ML Models (~2-5 min)

```bash
# Install XGBoost first (recommended)
pip install xgboost

# Train models
python src/train_ml_models.py
```

**What happens:**
- Trains XGBoost, Random Forest, SVR, ElasticNet
- Evaluates on test set
- Saves models and results
- Shows which model is best

---

### Step 3: Compare with CNN (Optional, <1 min)

```bash
python src/compare_ml_cnn.py
```

**What happens:**
- Compares ML vs CNN results
- Shows improvement
- Provides recommendations

---

## Expected Results

| Model | Expected Test R² | Status |
|-------|------------------|--------|
| CNN (current) | 0.199 | Your current result |
| **XGBoost** | **0.35-0.55** | ⭐ Expected best |
| Random Forest | 0.30-0.50 | Good |
| SVR | 0.25-0.40 | Baseline |
| Ensemble | **0.40-0.60** | Best possible |

---

## Output Files

After running:

```
features/               # Extracted features (Step 1)
models/                 # Trained ML models (Step 2)
results_ml/             # ML results and comparison (Step 2-3)
```

---

## Why This Works Better

1. ✅ **Small dataset (192 samples)** → Traditional ML > Deep Learning
2. ✅ **CNN features** → Better than hand-crafted features
3. ✅ **XGBoost** → Excellent for regression with limited data
4. ✅ **Fast training** → Minutes vs hours
5. ✅ **Standard approach** → Common in medical imaging research

---

## Next Steps

1. **Run Step 1** (extract features)
2. **Run Step 2** (train models)
3. **Check results** → Should see R² > 0.30
4. **Run Step 3** (compare with CNN)

**Ready to try?** Start with Step 1!

---

## Troubleshooting

**XGBoost not installed?**
```bash
pip install xgboost
```
(Other models will still work)

**Features not found?**
- Run Step 1 first: `python src/extract_features.py`

**Need more details?**
- See: `docs/ML_APPROACH_USAGE.md`
- See: `docs/CNN_FEATURES_ML_PLAN.md`

