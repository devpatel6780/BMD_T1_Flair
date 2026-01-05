# Testing Different Saved Models

This guide shows you how to test different saved models easily.

## Quick Method: Use test_model.py Helper Script

### List all available models:
```bash
python test_model.py --list
```

### Test a specific model:
```bash
# Test model from root
python test_model.py best_model.pth

# Test model from a folder
python test_model.py Modelresults1/best_model_Modelresults1.pth
python test_model.py full_finetune/best_model_full_finetune.pth
python test_model.py partial_finetune/best_model_partial_finetune.pth
```

This will automatically:
1. Update `src/config.py` to use the specified model
2. Set `TESTSCRIPT = True`
3. You then run `python train.py` to see results

## Manual Method: Edit config.py

1. Open `src/config.py`
2. Set:
   ```python
   TESTSCRIPT = True
   TEST_MODEL_PATH = "Modelresults1/best_model_Modelresults1.pth"  # Your model path
   ```
3. Run: `python train.py`

## Available Models (as of now)

Based on your project, you have these models:

1. `best_model.pth` - Root directory (SimpleCNN)
2. `Modelresults1/best_model_Modelresults1.pth`
3. `Modelresults2/best_model_Modelresults2.pth`
4. `full_finetune/best_model_full_finetune.pth` - ResNetTransfer (full fine-tuning)
5. `partial_finetune/best_model_partial_finetune.pth` - ResNetTransfer (partial fine-tuning)
6. `frozen_backbone/best_model_frozen_backbone.pth` - ResNetTransfer (frozen backbone)
7. `regularized_v1/best_model_regularized_v1.pth`

## Example Workflow

```bash
# 1. List all models
python test_model.py --list

# 2. Test Modelresults1
python test_model.py Modelresults1/best_model_Modelresults1.pth
python train.py

# 3. Test full_finetune
python test_model.py full_finetune/best_model_full_finetune.pth
python train.py

# 4. Compare results from different models
```

## Notes

- The script automatically detects model architecture (SimpleCNN vs ResNetTransfer)
- Results are shown as metrics only (no plots) when `TESTSCRIPT = True`
- Each model's results will show R², MAE, RMSE, Pearson correlation, etc.

