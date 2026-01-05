# Regression Project Outputs Guide

This document explains what outputs and visualizations are essential for regression projects, especially in medical imaging contexts.

## Essential Outputs for Regression Projects

### 1. **Metrics (Quantitative Evaluation)**

#### Primary Metrics:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
  - Interpretation: Lower is better, measured in same units as target
  - Example: MAE = 0.15 means average error is 0.15 BMD units

- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
  - Interpretation: Penalizes large errors more than MAE
  - Example: RMSE = 0.20 means typical error magnitude

- **R² (Coefficient of Determination)**: Proportion of variance explained
  - Range: -∞ to 1.0
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model performs as well as predicting the mean
  - R² < 0.0: Model performs worse than predicting the mean
  - **Goal**: R² > 0.5 is typically good, R² > 0.7 is excellent

- **Pearson Correlation**: Linear relationship strength
  - Range: -1.0 to 1.0
  - Interpretation: How well predictions track actual values
  - Example: 0.8 means strong positive correlation

#### Extended Metrics:
- **Median Absolute Error**: Robust to outliers
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
- **Error Statistics**: Min, Max, Q25, Q75, Std of errors
- **Distribution Statistics**: Mean and Std of actual vs predicted

---

### 2. **Visualizations (Qualitative Evaluation)**

#### A. **Predicted vs Actual Scatter Plot**
**Purpose**: Visual assessment of prediction accuracy

**What to look for**:
- Points should cluster along the diagonal line (y=x)
- Tight clustering = good predictions
- Systematic bias = points consistently above/below diagonal
- Outliers = points far from diagonal

**Includes**:
- Perfect prediction line (y=x) in red
- Linear fit line in blue
- Metrics text box (R², MAE, RMSE, Pearson)

**File**: `{split}_predicted_vs_actual.png`

---

#### B. **Residual Plots**
**Purpose**: Identify prediction patterns and biases

**Two panels**:
1. **Residuals vs Actual**: 
   - Should show random scatter around zero
   - Patterns indicate systematic errors
   - Funnel shape = heteroscedasticity (variance changes)

2. **Residual Distribution**:
   - Should be centered at zero
   - Normal distribution is ideal
   - Skewness indicates bias

**File**: `{split}_residuals.png`

---

#### C. **Error Distribution Analysis**
**Purpose**: Comprehensive error analysis

**Four panels**:
1. **Absolute Error Distribution**: Histogram of error magnitudes
2. **Distribution Comparison**: Overlay of actual vs predicted distributions
3. **Error vs Predicted**: Check for systematic errors by prediction value
4. **Q-Q Plot**: Check if residuals follow normal distribution

**File**: `{split}_error_distribution.png`

---

#### D. **Learning Curves**
**Purpose**: Monitor training progress and detect overfitting

**Panels**:
1. **Loss Curves**: Train vs Validation loss (linear scale)
2. **Loss Curves (Log Scale)**: Better visualization of early training
3. **Validation MAE**: Per-epoch validation MAE
4. **Validation R²**: Per-epoch validation R²

**What to look for**:
- Training loss should decrease steadily
- Validation loss should track training loss (no large gap = no overfitting)
- Validation metrics should improve over time
- Plateaus indicate convergence

**File**: `learning_curves.png`

---

### 3. **Data Files**

#### A. **Predictions CSV**
Contains per-sample predictions with:
- Patient ID (if available)
- Actual value
- Predicted value
- Error (Predicted - Actual)
- Absolute Error
- Percentage Error

**Use cases**:
- Detailed error analysis
- Identifying problematic samples
- Further statistical analysis

**File**: `{split}_predictions.csv`

---

#### B. **Metrics CSV**
Single-row CSV with all computed metrics:
- MSE, RMSE, MAE, Median_AE
- R², Pearson, MAPE
- Error statistics
- Distribution statistics

**Use cases**:
- Comparing different models
- Tracking improvements
- Reporting results

**File**: `{split}_metrics.csv`

---

## Output Structure

After running training, the `results/` directory will contain:

```
results/
├── test_predictions.csv          # Detailed test predictions
├── test_metrics.csv               # Test set metrics summary
├── test_predicted_vs_actual.png  # Test scatter plot
├── test_residuals.png            # Test residual analysis
├── test_error_distribution.png   # Test error analysis
├── validation_predictions.csv    # Validation predictions
├── validation_metrics.csv        # Validation metrics
├── validation_predicted_vs_actual.png
├── validation_residuals.png
├── validation_error_distribution.png
└── learning_curves.png           # Training history
```

---

## Interpretation Guide

### Good Regression Results:
✅ **Scatter Plot**: Points tightly clustered along diagonal
✅ **R² > 0.5**: Model explains >50% of variance
✅ **Pearson > 0.7**: Strong linear relationship
✅ **Residuals**: Random scatter, centered at zero
✅ **Learning Curves**: Steady improvement, no overfitting

### Poor Regression Results:
❌ **Scatter Plot**: Points scattered randomly or in horizontal band
❌ **R² < 0**: Model worse than predicting mean
❌ **Residuals**: Clear patterns (curves, funnels)
❌ **Predictions clustered**: All predictions similar (model not learning)
❌ **Large gap**: Train loss << Val loss (overfitting)

---

## Medical Imaging Specific Considerations

### Patient-Level Evaluation:
- **Why**: Multiple scans per patient should be aggregated
- **How**: Average predictions across all scans per patient
- **Benefit**: More clinically relevant, prevents data leakage

### Per-Patient Metrics:
- All metrics computed after aggregating scan-level predictions
- More stable and reliable than scan-level metrics
- Better reflects real-world clinical usage

### Error Analysis:
- Identify patients with consistently high errors
- May indicate:
  - Image quality issues
  - Annotation errors
  - Edge cases requiring special handling

---

## Example Output Summary

```
============================================================
         Extended Regression Metrics (TEST)
============================================================
MSE              : 0.041600
RMSE             : 0.204000
MAE              : 0.168500
Median AE        : 0.152300
R²               : 0.320000
Pearson Corr     : 0.565700
MAPE (%)         : 18.4500

Error Statistics:
  Std            : 0.185000
  Min            : -0.450000
  Max            : 0.380000
  Q25            : -0.120000
  Q75            : 0.150000

Distribution Statistics:
  Actual Mean    : 0.950000
  Predicted Mean : 0.920000
  Actual Std     : 0.280000
  Predicted Std  : 0.220000
============================================================
```

---

## Best Practices

1. **Always generate visualizations**: Numbers alone don't tell the full story
2. **Compare distributions**: Actual vs predicted should have similar distributions
3. **Check residuals**: Patterns indicate model limitations
4. **Monitor learning curves**: Early stopping if validation plateaus
5. **Save all outputs**: For reproducibility and comparison
6. **Report per-patient metrics**: More meaningful for medical applications
7. **Include confidence intervals**: If possible, show prediction uncertainty

---

## Quick Checklist

Before considering a regression project complete, ensure you have:

- [ ] Scatter plot (predicted vs actual)
- [ ] Residual plots
- [ ] Error distribution analysis
- [ ] Learning curves (if training)
- [ ] CSV with all predictions
- [ ] CSV with all metrics
- [ ] R², MAE, RMSE, Pearson correlation reported
- [ ] Patient-level evaluation (if applicable)
- [ ] Error statistics (min, max, percentiles)
- [ ] Distribution comparison (actual vs predicted)

---

## Additional Resources

- **Scikit-learn metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- **Regression diagnostics**: https://en.wikipedia.org/wiki/Regression_diagnosis
- **Medical imaging evaluation**: Consider clinical significance, not just statistical metrics

