# Regression Visualization Implementation Summary

## What Was Added

I've implemented a comprehensive regression results visualization and reporting system for your BMD prediction project.

## New Files Created

### 1. `src/visualize_results.py`
A complete visualization module with functions for:
- Extended regression metrics (MAE, RMSE, R², Pearson, MAPE, etc.)
- Scatter plots (predicted vs actual)
- Residual analysis plots
- Error distribution analysis
- Learning curves visualization
- CSV export of predictions and metrics

### 2. `docs/REGRESSION_OUTPUTS_GUIDE.md`
Comprehensive guide explaining:
- What outputs are needed for regression projects
- How to interpret each visualization
- Best practices for regression evaluation
- Medical imaging specific considerations

## Modified Files

### `train.py`
Updated to:
- Import and use the visualization module
- Track training/validation metrics during training
- Generate comprehensive reports for test and validation sets
- Create learning curves automatically
- Save all outputs to `results/` directory

## What You Get Now

When you run `python train.py`, the system will automatically generate:

### Visualizations (PNG files):
1. **Predicted vs Actual Scatter Plot**: Shows how well predictions match actual values
2. **Residual Plots**: Analyzes prediction errors
3. **Error Distribution**: Comprehensive error analysis
4. **Learning Curves**: Training progress visualization

### Data Files (CSV):
1. **Predictions CSV**: Detailed per-patient predictions with errors
2. **Metrics CSV**: Summary of all computed metrics

### Output Location:
All files are saved to `results/` directory:
```
results/
├── test_predictions.csv
├── test_metrics.csv
├── test_predicted_vs_actual.png
├── test_residuals.png
├── test_error_distribution.png
├── validation_predictions.csv
├── validation_metrics.csv
├── validation_predicted_vs_actual.png
├── validation_residuals.png
├── validation_error_distribution.png
└── learning_curves.png
```

## Key Features

### Extended Metrics:
- Basic: MAE, RMSE, R², Pearson
- Advanced: Median AE, MAPE, Error statistics (min, max, Q25, Q75)
- Distribution: Mean and Std of actual vs predicted

### Visualizations:
- **Scatter plots** with perfect prediction line and linear fit
- **Residual analysis** to detect systematic errors
- **Error distributions** to understand prediction patterns
- **Learning curves** to monitor training progress

### Patient-Level Evaluation:
- All metrics computed per-patient (after aggregating scan predictions)
- More clinically relevant for medical imaging
- Prevents data leakage issues

## Usage

Simply run your training script as usual:
```bash
python train.py
```

The visualization system will automatically:
1. Track metrics during training
2. Generate comprehensive reports after training
3. Save all outputs to `results/` directory

## Dependencies

The visualization module requires:
- `numpy`
- `matplotlib`
- `pandas`
- `scipy` (optional, for Q-Q plots)

If scipy is not available, Q-Q plots will show a message instead.

## Example Output

After training, you'll see:
- Console output with extended metrics
- All visualizations saved as high-resolution PNG files
- CSV files for further analysis
- Learning curves showing training progress

## Next Steps

1. Run training to generate the reports
2. Review the visualizations to understand model performance
3. Use the CSV files for detailed error analysis
4. Compare metrics across different model configurations

## Benefits

✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
✅ **Easy Interpretation**: Visual plots make results clear
✅ **Reproducibility**: All outputs saved for comparison
✅ **Professional Output**: Publication-ready visualizations
✅ **Error Analysis**: Identify problematic samples
✅ **Training Monitoring**: Learning curves show training progress

