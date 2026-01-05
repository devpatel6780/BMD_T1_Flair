# Model Naming and Run Management Guide

## Overview

The system now supports saving different model checkpoints and results for each training run, preventing overwriting of previous experiments.

## How It Works

### Automatic Timestamp-Based Naming (Default)

When `MODEL_NAME = None` in `config.py`:
- **Model file**: `best_model_YYYYMMDD_HHMMSS.pth`
  - Example: `best_model_20241215_143022.pth`
- **Results directory**: `results_YYYYMMDD_HHMMSS/`
  - Example: `results_20241215_143022/`

Each run gets a unique name based on when training started.

### Custom Naming

When you set `MODEL_NAME = "your_name"` in `config.py`:
- **Model file**: `best_model_your_name.pth`
  - Example: `best_model_baseline.pth`
- **Results directory**: `results_your_name/`
  - Example: `results_baseline/`

## Usage Examples

### Example 1: Auto-Generate Names (Recommended for Experiments)

In `src/config.py`:
```python
MODEL_NAME = None  # Auto-generate timestamp-based name
```

**Result:**
- Model: `best_model_20241215_143022.pth`
- Results: `results_20241215_143022/`

### Example 2: Custom Name for Specific Experiment

In `src/config.py`:
```python
MODEL_NAME = "baseline_v1"
```

**Result:**
- Model: `best_model_baseline_v1.pth`
- Results: `results_baseline_v1/`

### Example 3: Track Different Configurations

```python
# Run 1: Baseline
MODEL_NAME = "baseline"

# Run 2: With augmentation
MODEL_NAME = "with_augmentation"

# Run 3: Different learning rate
MODEL_NAME = "lr_1e3"

# Run 4: Different architecture
MODEL_NAME = "deeper_cnn"
```

## File Structure After Multiple Runs

```
BMD_T1_Flair-main/
├── best_model_20241215_143022.pth    # Run 1 (auto-named)
├── best_model_baseline.pth            # Run 2 (custom name)
├── best_model_with_aug.pth            # Run 3 (custom name)
├── results_20241215_143022/          # Results for Run 1
│   ├── test_predictions.csv
│   ├── test_metrics.csv
│   ├── test_predicted_vs_actual.png
│   └── ...
├── results_baseline/                   # Results for Run 2
│   └── ...
└── results_with_aug/                  # Results for Run 3
    └── ...
```

## Best Practices

### 1. **Use Descriptive Names**
```python
MODEL_NAME = "n_slices_5_lr_2e4"  # Describes key hyperparameters
MODEL_NAME = "augmented_data_v2"  # Describes data changes
MODEL_NAME = "transfer_learning"  # Describes approach
```

### 2. **Keep a Log**
Create a simple text file or spreadsheet tracking:
- Model name
- Key hyperparameters
- Results (R², MAE, etc.)
- Notes

### 3. **Organize by Experiment Type**
```python
# Baseline experiments
MODEL_NAME = "baseline_run1"
MODEL_NAME = "baseline_run2"

# Augmentation experiments
MODEL_NAME = "aug_rotation"
MODEL_NAME = "aug_flip"

# Architecture experiments
MODEL_NAME = "deeper_cnn"
MODEL_NAME = "wider_cnn"
```

## Configuration File Location

Edit: `src/config.py`

```python
# Model saving configuration
MODEL_NAME = None  # Change this to customize

# The rest is automatic:
# - BEST_PATH is set based on MODEL_NAME
# - RESULTS_DIR is set based on MODEL_NAME
```

## What Gets Saved

### Model Checkpoint (`best_model_*.pth`)
- Model state dict
- Normalization stats (y_mean, y_std)
- Model configuration (in_ch, n_slices, etc.)

### Results Directory (`results_*/`)
- Test predictions CSV
- Test metrics CSV
- Validation predictions CSV
- Validation metrics CSV
- All visualization plots (PNG)
- Learning curves

## Tips

1. **Quick Comparison**: Use custom names to easily compare specific experiments
2. **Reproducibility**: Timestamp names help track when experiments were run
3. **Version Control**: Consider adding model names to your experiment log
4. **Storage**: Old models can be archived or deleted if storage is limited

## Example Workflow

```python
# Day 1: Baseline experiment
MODEL_NAME = "baseline"
# Run training → saves to best_model_baseline.pth

# Day 2: Try different learning rate
MODEL_NAME = "lr_1e3"
# Run training → saves to best_model_lr_1e3.pth

# Day 3: Compare with augmentation
MODEL_NAME = "with_aug"
# Run training → saves to best_model_with_aug.pth

# Compare results from results_baseline/, results_lr_1e3/, results_with_aug/
```

## Answer to Your Question

**Yes, your model is saved in `best_model.pth`** (or the configured name).

Now with this update:
- ✅ Each run can have a unique model name
- ✅ Results are organized by run
- ✅ No overwriting of previous experiments
- ✅ Easy to compare different runs

Just change `MODEL_NAME` in `config.py` before each run!

