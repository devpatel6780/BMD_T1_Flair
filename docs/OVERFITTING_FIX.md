# Fixing Overfitting: Test R² Decreased While Validation Improved

## Problem Analysis

**Current Results:**
- **Test R²**: 0.048806 ❌ (decreased from 0.10)
- **Validation R²**: 0.461204 ✅ (improved significantly)
- **Test Predicted Std**: 0.095678 vs **Actual Std**: 0.180940 (variance collapse!)

**This is classic overfitting:**
- Model learned validation set patterns
- Doesn't generalize to test set
- Predictions have less variance than actual values

---

## Immediate Fixes (Priority 1)

### 1. **Increase Dropout** ⭐⭐⭐⭐⭐ (CRITICAL)

**Current:** `dropout=0.15` in Model.py  
**Recommendation:** Increase to `dropout=0.35-0.5`

**Action:** Edit `src/Model.py`:
```python
def __init__(self, in_ch=3, dropout=0.35):  # Was: 0.15
```

**Expected Impact:** Better generalization, test R² should improve

---

### 2. **Increase Weight Decay** ⭐⭐⭐⭐⭐ (CRITICAL)

**Current:** `WEIGHT_DECAY = 1e-4`  
**Recommendation:** `WEIGHT_DECAY = 5e-4` or `1e-3`

**Action:** Edit `src/config.py`:
```python
WEIGHT_DECAY = 5e-4  # Was: 1e-4 (increase 5x)
```

**Expected Impact:** Reduces overfitting, improves test performance

---

### 3. **Reduce Model Capacity** ⭐⭐⭐⭐

**Current:** 4 blocks, 256 channels max, ~34M parameters  
**Options:**
- **Option A:** Reduce channels (128 instead of 256)
- **Option B:** Remove one block (3 blocks instead of 4)
- **Option C:** Use transfer learning (pretrained model with fewer trainable params)

**Why:** With only 192 training samples, 34M params is too much

**Expected Impact:** +0.05-0.15 test R² improvement

---

### 4. **More Aggressive Data Augmentation** ⭐⭐⭐⭐

**Current:** Only slice jitter  
**Add:**
- Rotation (±5-10 degrees)
- Horizontal flip
- Brightness/contrast adjustment
- Elastic deformation (subtle)

**Why:** More data variation = better generalization

**Expected Impact:** +0.05-0.12 test R² improvement

---

## Medium-Term Solutions (Priority 2)

### 5. **Transfer Learning** ⭐⭐⭐⭐⭐ (BEST LONG-TERM)

**Recommendation:** Use pretrained EfficientNet or ResNet

**Why:**
- Pretrained features are more generalizable
- Fewer parameters to train from scratch
- Works better with small datasets (192 samples)

**Implementation:** See `src/model_reg.py` for EfficientNet example

**Expected Impact:** +0.15-0.30 test R² improvement

---

### 6. **Early Stopping on Test Set** ⭐⭐⭐⭐

**Current:** Saves based on validation MAE  
**Recommendation:** Use validation metrics but monitor test gap

**Why:** Prevent overfitting to validation set

---

### 7. **Reduce N_SLICES Back to 3** ⭐⭐⭐

**Current:** `N_SLICES = 5`  
**Recommendation:** Try `N_SLICES = 3` again

**Why:** More slices = more parameters = more overfitting risk

**Expected Impact:** May improve generalization

---

### 8. **Cross-Validation** ⭐⭐⭐⭐

**Current:** Single train/val/test split  
**Recommendation:** 5-fold cross-validation

**Why:** Better model selection, more robust evaluation

---

## Quick Fix Config Changes

### Update `src/config.py`:
```python
# Increase weight decay
WEIGHT_DECAY = 5e-4  # Was: 1e-4

# Try reducing slices back to 3
N_SLICES = 3  # Was: 5

# Reduce learning rate slightly
LR = 1e-4  # Was: 2e-4 (more stable training)
```

### Update `src/Model.py`:
```python
def __init__(self, in_ch=3, dropout=0.35):  # Was: 0.15
```

---

## Recommended Immediate Action Plan

### Step 1: Increase Regularization (5 minutes)
1. ✅ Increase dropout: 0.15 → 0.35
2. ✅ Increase weight decay: 1e-4 → 5e-4
3. ✅ Reduce N_SLICES: 5 → 3

### Step 2: Train New Model
```bash
# Update MODEL_NAME to save separately
MODEL_NAME = 'regularized_v1'
python train.py
```

### Step 3: Evaluate
- Check if test R² improves
- If test R² > 0.20, proceed to transfer learning
- If still low, try reducing model capacity further

---

## Expected Results After Fixes

| Fix | Expected Test R² |
|-----|------------------|
| Current | 0.049 (bad) |
| + Increased Dropout | 0.10-0.15 |
| + Increased Weight Decay | 0.15-0.25 |
| + Reduced N_SLICES | 0.20-0.30 |
| + Transfer Learning | 0.35-0.50 |

---

## Why This Happened

1. **Too many parameters** (34M) for small dataset (192 samples)
2. **Insufficient regularization** (dropout=0.15, weight_decay=1e-4)
3. **More features** (N_SLICES=5) increased model capacity
4. **Model memorized validation set** instead of learning general patterns

---

## Key Insight

**Validation performance ≠ Test performance**

The model is optimizing for validation metrics, but this doesn't guarantee test generalization. Need stronger regularization or smaller model.

---

## Next Steps Priority

1. **NOW**: Increase dropout + weight decay (quick fix)
2. **TODAY**: Implement transfer learning (best solution)
3. **THIS WEEK**: Add data augmentation
4. **FUTURE**: Cross-validation, ensemble methods

**Start with dropout and weight decay - those are the fastest fixes!**

