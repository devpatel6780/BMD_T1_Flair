# Test Script Guide

## Overview

The test script mode allows you to quickly evaluate a saved model without training. It loads the model and shows only the results/metrics (no plots or CSV files).

## How to Use

### Step 1: Set MODEL_NAME in config.py

Make sure `MODEL_NAME` matches the folder name where your saved model is located:

```python
# In src/config.py
MODEL_NAME = 'Modelresults1'  # Must match the folder containing your model
```

### Step 2: Enable Test Mode

Set `TESTSCRIPT = True` in `src/config.py`:

```python
# In src/config.py
TESTSCRIPT = True  # Enable test-only mode
```

### Step 3: Run

```bash
python train.py
```

## What Happens

When `TESTSCRIPT = True`:

1. ✅ **Skips training** - No model training occurs
2. ✅ **Loads saved model** - Loads from `{MODEL_NAME}/best_model_{MODEL_NAME}.pth`
3. ✅ **Evaluates on test set** - Shows metrics only
4. ✅ **Evaluates on validation set** - Shows metrics only
5. ❌ **No plots generated** - Only console output
6. ❌ **No CSV files saved** - Only console output

## Example Output

```
======================================================================
TEST-ONLY MODE: Loading saved model and evaluating
======================================================================
Model name: Modelresults1
Loading model from: Modelresults1/best_model_Modelresults1.pth
Model loaded successfully!
Normalization stats: mean=0.950000, std=0.280000
======================================================================

Evaluating on TEST set...

======================================================================
         TEST Results (Per-Patient) - Metrics Only
======================================================================
MSE              : 0.018586
RMSE             : 0.136330
MAE              : 0.105768
Median AE        : 0.093157
R²               : 0.368222
Pearson Corr     : 0.609281
MAPE (%)         : 11.5774

Error Statistics:
  Std            : 0.136066
  Min            : -0.300785
  Max            : 0.273430
  Q25            : -0.090291
  Q75            : 0.083298

Distribution Statistics:
  Actual Mean    : 0.950000
  Predicted Mean : 0.920000
  Actual Std     : 0.280000
  Predicted Std  : 0.220000
======================================================================

Prediction Examples (first 10 patients):
--------------------------------------------------
 Patient ID  |   Actual    |  Predicted   |    Error    
--------------------------------------------------
    123      |    1.280    |    1.150     |   -0.130    
    456      |    0.931    |    0.980     |    0.049    
    ...
--------------------------------------------------
```

## Important Notes

1. **MODEL_NAME must match**: The `MODEL_NAME` in config must match the folder name containing your saved model
2. **Model file location**: The script looks for: `{MODEL_NAME}/best_model_{MODEL_NAME}.pth`
3. **No training**: Training is completely skipped when `TESTSCRIPT = True`
4. **Results only**: Only metrics are shown, no visualizations or files are generated

## Troubleshooting

### Error: Model file not found

```
FileNotFoundError: Model file not found: Modelresults1/best_model_Modelresults1.pth
```

**Solution**: 
- Check that `MODEL_NAME` in config matches your folder name
- Verify the model file exists in that folder
- Make sure the folder name matches exactly (case-sensitive)

### Error: Channel mismatch

```
Warning: Config N_SLICES=3 (in_ch=3) but checkpoint has in_ch=5
```

**Solution**: 
- The script will use the checkpoint's value automatically
- Or update `N_SLICES` in config to match the saved model

## Quick Reference

| Config Setting | Value | Description |
|---------------|-------|-------------|
| `TESTSCRIPT` | `True` | Enable test-only mode |
| `TESTSCRIPT` | `False` | Normal training mode |
| `MODEL_NAME` | `'Modelresults1'` | Folder name containing saved model |

## Workflow Example

```python
# 1. Train a model (TESTSCRIPT = False)
MODEL_NAME = 'experiment_1'
TESTSCRIPT = False
# Run: python train.py
# → Saves: experiment_1/best_model_experiment_1.pth

# 2. Later, test the model (TESTSCRIPT = True)
MODEL_NAME = 'experiment_1'  # Same name!
TESTSCRIPT = True
# Run: python train.py
# → Loads and evaluates, shows results only
```

