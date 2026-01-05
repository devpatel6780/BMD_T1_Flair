# Model Performance Improvement Guide

## Current Results Analysis

**Test Set Metrics:**
- R²: 0.104898 (Very Low - Goal: >0.5)
- MAE: 0.130998
- RMSE: 0.171187
- Pearson Correlation: 0.380920 (Low - Goal: >0.7)

**Key Issues:**
1. **Low R²** (0.10) - Model explains only 10% of variance
2. **Low correlation** (0.38) - Weak linear relationship
3. **Limited data** (240 samples total, 192 train)
4. **Small training epochs** (2 epochs - too few!)

---

## Prioritized Improvement Strategy

### 🚀 **Priority 1: Quick Wins (High Impact, Easy Implementation)**

#### 1.1 **Increase Training Epochs** ⭐⭐⭐⭐⭐
**Current:** `EPOCHS = 2`  
**Recommendation:** `EPOCHS = 100-150`

**Why:** Only 2 epochs is far too few for the model to learn. Medical imaging models typically need 50-200 epochs.

**Action:**
```python
# In src/config.py
EPOCHS = 100  # Start with 100, can go up to 150-200
```

**Expected Impact:** +0.15-0.25 R² improvement

---

#### 1.2 **Enable Robust Normalization** ⭐⭐⭐⭐⭐
**Current:** `USE_ROBUST_NORM = False`  
**Recommendation:** `USE_ROBUST_NORM = True`

**Why:** MRI images often have non-Gaussian intensity distributions. Robust normalization (median/IQR) is more stable than z-score for medical images.

**Action:**
```python
# In src/config.py
USE_ROBUST_NORM = True  # Better for MRI images
```

**Expected Impact:** +0.05-0.10 R² improvement

---

#### 1.3 **Increase Batch Size (if memory allows)** ⭐⭐⭐⭐
**Current:** `BATCH_SIZE = 8`  
**Recommendation:** `BATCH_SIZE = 16` or `32`

**Why:** Larger batches provide more stable gradients and often lead to better generalization.

**Action:**
```python
# In src/config.py
BATCH_SIZE = 16  # If GPU memory allows, try 32
```

**Expected Impact:** +0.03-0.08 R² improvement

---

#### 1.4 **Increase Number of Slices** ⭐⭐⭐⭐
**Current:** `N_SLICES = 3`  
**Recommendation:** Try `N_SLICES = 5` or `7`

**Why:** More slices provide richer spatial context, which may help for BMD prediction.

**Action:**
```python
# In src/config.py
N_SLICES = 5  # Try 5 or 7 (odd numbers work best)
```

**Expected Impact:** +0.05-0.15 R² improvement (depends on data)

**Note:** This increases model input channels, so verify model compatibility.

---

### 🎯 **Priority 2: Data Augmentation (High Impact, Medium Effort)**

#### 2.1 **Add Image Augmentation**
**Current:** Only slice jitter  
**Recommendation:** Add rotation, flipping, elastic deformation

**Implementation Ideas:**
- **Rotation**: ±5-10 degrees (spine is mostly vertical)
- **Horizontal Flip**: Left-right (anatomical symmetry)
- **Brightness/Contrast**: ±10-15% (MRI intensity variations)
- **Elastic Deformation**: Subtle deformations

**Expected Impact:** +0.10-0.20 R² improvement

---

#### 2.2 **Increase Slice Jitter**
**Current:** `SLICE_JITTER = 1`  
**Recommendation:** `SLICE_JITTER = 2` or `3`

**Why:** More slice variation helps model generalize to different slice positions.

**Action:**
```python
# In src/config.py
SLICE_JITTER = 2  # Increase from 1 to 2 or 3
```

**Expected Impact:** +0.03-0.08 R² improvement

---

### 🔧 **Priority 3: Model Architecture Improvements**

#### 3.1 **Transfer Learning with Pretrained Models** ⭐⭐⭐⭐⭐
**Current:** Training from scratch  
**Recommendation:** Use pretrained ResNet, EfficientNet, or DenseNet

**Why:** 
- Medical imaging datasets are small (240 samples)
- Pretrained models on ImageNet provide useful features
- Can fine-tune last layers for regression

**Expected Impact:** +0.15-0.30 R² improvement

**Implementation:** See `src/model_reg.py` for EfficientNet-based example

---

#### 3.2 **Increase Model Depth/Capacity** ⭐⭐⭐
**Current:** 4 blocks, 256 channels max  
**Recommendation:** Add more layers or increase channels

**Action:** Modify `src/Model.py` to add:
- Additional convolutional blocks
- More channels (e.g., 512 instead of 256)
- Attention mechanisms (see `src/model_reg.py`)

**Expected Impact:** +0.05-0.15 R² improvement (but risk of overfitting)

---

#### 3.3 **Add Attention Mechanisms** ⭐⭐⭐⭐
**Current:** Standard CNN  
**Recommendation:** Add SE (Squeeze-and-Excitation) blocks or CBAM

**Why:** Attention helps model focus on relevant regions (spine area).

**Implementation:** Examples in `src/model_reg.py` (ECA, cSE, sSE blocks)

**Expected Impact:** +0.05-0.12 R² improvement

---

### 📊 **Priority 4: Training Strategy**

#### 4.1 **Learning Rate Scheduling** ⭐⭐⭐⭐
**Current:** ReduceLROnPlateau (good, but can improve)

**Recommendation:** Try:
- **Cosine Annealing**: Often better for longer training
- **Warm Restarts**: Help escape local minima
- **One Cycle Policy**: Popular for medical imaging

**Expected Impact:** +0.03-0.10 R² improvement

---

#### 4.2 **Loss Function Tuning** ⭐⭐⭐
**Current:** SmoothL1Loss (beta=0.5)  
**Recommendation:** Experiment with:
- **Huber Loss** with different beta values (0.25, 0.5, 1.0)
- **Combined Loss**: MSE + Pearson Correlation (see `src/model_reg.py`)
- **Quantile Loss**: For robust regression

**Expected Impact:** +0.02-0.08 R² improvement

---

#### 4.3 **Early Stopping with Patience** ⭐⭐⭐⭐
**Current:** Training for fixed epochs  
**Recommendation:** Implement early stopping based on validation MAE

**Why:** Prevents overfitting and saves training time.

**Expected Impact:** Prevents overfitting, maintains generalization

---

### 🎲 **Priority 5: Advanced Techniques**

#### 5.1 **Ensemble Methods** ⭐⭐⭐⭐⭐
**Recommendation:** Train multiple models and average predictions

**Implementation:**
- Train 3-5 models with different random seeds
- Average their predictions at test time
- Can combine different architectures

**Expected Impact:** +0.05-0.15 R² improvement

---

#### 5.2 **Cross-Validation** ⭐⭐⭐⭐
**Current:** Single train/val/test split  
**Recommendation:** 5-fold cross-validation

**Why:** Better utilizes limited data and provides more reliable metrics.

**Expected Impact:** More robust model selection

---

#### 5.3 **Test-Time Augmentation (TTA)** ⭐⭐⭐⭐
**Recommendation:** Average predictions over multiple augmentations at test time

**Implementation:**
- Generate predictions with original + flipped + rotated images
- Average the predictions

**Expected Impact:** +0.02-0.08 R² improvement

---

#### 5.4 **Focal Loss or Uncertainty Estimation** ⭐⭐⭐
**Recommendation:** Weight difficult samples more or estimate prediction uncertainty

**Why:** Helps model focus on hard cases and provides confidence intervals.

---

## Recommended Implementation Order

### Phase 1: Quick Fixes (1-2 hours)
1. ✅ Increase EPOCHS to 100
2. ✅ Enable USE_ROBUST_NORM = True
3. ✅ Increase BATCH_SIZE to 16
4. ✅ Increase SLICE_JITTER to 2

**Expected R²:** 0.10 → 0.25-0.35

---

### Phase 2: Data Improvements (2-4 hours)
1. Add image augmentation (rotation, flip, brightness)
2. Try N_SLICES = 5
3. Implement TTA

**Expected R²:** 0.25-0.35 → 0.35-0.50

---

### Phase 3: Architecture Improvements (4-8 hours)
1. Implement transfer learning (EfficientNet/ResNet)
2. Add attention mechanisms
3. Try ensemble methods

**Expected R²:** 0.35-0.50 → 0.50-0.70

---

### Phase 4: Advanced (8+ hours)
1. Cross-validation
2. Advanced loss functions
3. Hyperparameter optimization

**Expected R²:** 0.50-0.70 → 0.65-0.80+

---

## Immediate Action Plan

### Step 1: Update Config (5 minutes)
```python
# src/config.py
EPOCHS = 100  # Was: 2
USE_ROBUST_NORM = True  # Was: False
BATCH_SIZE = 16  # Was: 8 (if GPU memory allows)
SLICE_JITTER = 2  # Was: 1
```

### Step 2: Train New Model
```bash
# Set TESTSCRIPT = False
python train.py
```

### Step 3: Evaluate
- Check if R² improved to 0.25-0.35
- If yes, proceed to Phase 2
- If no, review data quality/preprocessing

---

## Key Metrics to Track

**Primary Metrics:**
- **R²**: Goal >0.5 (good), >0.7 (excellent)
- **Pearson Correlation**: Goal >0.7 (good), >0.8 (excellent)
- **MAE**: Lower is better

**Monitoring:**
- Training vs Validation loss gap (overfitting indicator)
- Learning curves (should improve steadily)
- Error distribution (should be centered at zero)

---

## Common Pitfalls to Avoid

1. **Overfitting**: With only 192 training samples, be careful with large models
2. **Too few epochs**: Medical imaging needs sufficient training time
3. **Inconsistent preprocessing**: Ensure train/val/test use same preprocessing
4. **Data leakage**: Already handled (patient-level splits) ✅
5. **Ignoring validation metrics**: Monitor validation MAE/R², not just loss

---

## Expected Timeline

| Phase | Time | Expected R² |
|-------|------|-------------|
| Current | - | 0.10 |
| Phase 1 | 1-2 hours | 0.25-0.35 |
| Phase 2 | 2-4 hours | 0.35-0.50 |
| Phase 3 | 4-8 hours | 0.50-0.70 |
| Phase 4 | 8+ hours | 0.65-0.80+ |

---

## Research Context

For medical imaging regression tasks, typical R² values:
- **Poor**: <0.3
- **Acceptable**: 0.3-0.5
- **Good**: 0.5-0.7
- **Excellent**: >0.7

**Target for publication:** R² >0.5, preferably >0.6

---

## Questions to Consider

1. **Domain knowledge**: Are there specific anatomical features that correlate with BMD?
2. **Data quality**: Are all scans of similar quality? Any outliers?
3. **Ground truth**: How reliable are BMD labels? Any measurement errors?
4. **Multi-modal**: Can you combine T1 FLAIR with other MRI sequences?

---

## Next Steps

1. ✅ Start with Phase 1 (quick fixes)
2. ✅ Train and evaluate
3. ✅ If improvement is seen, proceed to Phase 2
4. ✅ Consider transfer learning if Phase 1 doesn't improve enough
5. ✅ Document all experiments and hyperparameters

**Good luck with your research!** 🚀

