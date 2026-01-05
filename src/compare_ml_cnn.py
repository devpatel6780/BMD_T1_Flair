"""
Compare ML models (trained on CNN features) with CNN end-to-end results.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import glob


def load_cnn_results(cnn_results_dir):
    """Load CNN test results from results directory"""
    cnn_dir = Path(cnn_results_dir)
    
    if not cnn_dir.exists():
        print(f"Warning: CNN results directory not found: {cnn_dir}")
        return None
    
    # Look for test_metrics.csv
    metrics_file = cnn_dir / "test_metrics.csv"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
        return metrics
    
    # If not found, try to find any results directory
    print(f"Warning: test_metrics.csv not found in {cnn_dir}")
    return None


def load_ml_results(ml_results_dir="results_ml"):
    """Load ML models comparison results"""
    ml_dir = Path(ml_results_dir)
    
    if not ml_dir.exists():
        print(f"Warning: ML results directory not found: {ml_dir}")
        return None
    
    comparison_file = ml_dir / "ml_models_comparison.csv"
    if comparison_file.exists():
        return pd.read_csv(comparison_file)
    
    return None


def find_best_cnn_result():
    """Find best CNN result from available result directories"""
    # Look for common result directories
    possible_dirs = [
        "full_finetune",
        "regularized_v1",
        "Modelresults1",
        "results",
    ]
    
    for dirname in possible_dirs:
        result_dir = Path(dirname)
        if result_dir.exists():
            metrics = load_cnn_results(result_dir)
            if metrics is not None:
                return metrics, dirname
    
    # Try to find any directory with test_metrics.csv
    for result_dir in Path(".").glob("*/test_metrics.csv"):
        parent_dir = result_dir.parent
        metrics = load_cnn_results(parent_dir)
        if metrics is not None:
            return metrics, str(parent_dir)
    
    return None, None


def main():
    print("=" * 70)
    print("CNN vs ML Models Comparison")
    print("=" * 70)
    
    # Load CNN results
    print("\nLoading CNN results...")
    cnn_metrics, cnn_dir = find_best_cnn_result()
    
    if cnn_metrics is None:
        print("Could not find CNN results. Please train CNN first or specify directory.")
        print("Looking for directories with test_metrics.csv")
        return
    
    print(f"Found CNN results in: {cnn_dir}")
    
    # Load ML results
    print("\nLoading ML results...")
    ml_df = load_ml_results()
    
    if ml_df is None:
        print("Could not find ML results. Please run train_ml_models.py first.")
        return
    
    # Create comparison
    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    
    comparison_data = []
    
    # Add CNN result
    comparison_data.append({
        'Model': 'CNN (End-to-End)',
        'R2': cnn_metrics.get('R2', 'N/A'),
        'MAE': cnn_metrics.get('MAE', 'N/A'),
        'RMSE': cnn_metrics.get('RMSE', 'N/A'),
        'Pearson': cnn_metrics.get('Pearson', 'N/A'),
        'Source': cnn_dir
    })
    
    # Add ML results
    for _, row in ml_df.iterrows():
        comparison_data.append({
            'Model': row['Model'],
            'R2': row['R2'],
            'MAE': row['MAE'],
            'RMSE': row['RMSE'],
            'Pearson': row.get('Pearson', 'N/A'),
            'Source': 'ML (CNN Features)'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison
    print("\nTest Set Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<25} | {'R²':>8} | {'MAE':>8} | {'RMSE':>8} | {'Pearson':>8}")
    print("-" * 70)
    
    for _, row in comparison_df.iterrows():
        r2 = row['R2']
        mae = row['MAE']
        rmse = row['RMSE']
        pearson = row['Pearson']
        
        if isinstance(r2, (int, float)):
            r2_str = f"{r2:.4f}"
        else:
            r2_str = str(r2)
        
        if isinstance(mae, (int, float)):
            mae_str = f"{mae:.4f}"
        else:
            mae_str = str(mae)
        
        if isinstance(rmse, (int, float)):
            rmse_str = f"{rmse:.4f}"
        else:
            rmse_str = str(rmse)
        
        if isinstance(pearson, (int, float)):
            pearson_str = f"{pearson:.4f}"
        else:
            pearson_str = str(pearson)
        
        print(f"{row['Model']:<25} | {r2_str:>8} | {mae_str:>8} | {rmse_str:>8} | {pearson_str:>8}")
    
    print("-" * 70)
    
    # Find best model
    numeric_df = comparison_df.copy()
    for col in ['R2', 'MAE', 'RMSE', 'Pearson']:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    best_idx = numeric_df['R2'].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_r2 = numeric_df.loc[best_idx, 'R2']
    
    print(f"\nBest Model (by R²): {best_model}")
    print(f"Best R²: {best_r2:.4f}")
    
    # Improvement over CNN
    cnn_r2 = numeric_df[numeric_df['Model'] == 'CNN (End-to-End)']['R2'].values[0]
    if not np.isnan(cnn_r2) and not np.isnan(best_r2):
        improvement = best_r2 - cnn_r2
        improvement_pct = (improvement / abs(cnn_r2)) * 100 if cnn_r2 != 0 else 0
        
        print(f"\nImprovement over CNN:")
        print(f"  Absolute: {improvement:+.4f}")
        print(f"  Relative: {improvement_pct:+.1f}%")
        
        if improvement > 0:
            print(f"\n✅ ML approach is BETTER than CNN!")
        elif improvement < 0:
            print(f"\n⚠️  CNN is still better, but ML might improve with tuning")
        else:
            print(f"\n➡️  Similar performance")
    
    # Save comparison
    output_dir = Path("results_ml")
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / "cnn_vs_ml_comparison.csv", index=False)
    
    print(f"\nComparison saved to: {output_dir}/cnn_vs_ml_comparison.csv")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    if best_r2 > 0.40:
        print("✅ Excellent results! R² > 0.40 is publication-quality.")
    elif best_r2 > 0.30:
        print("✅ Good results! R² > 0.30 is solid for medical imaging regression.")
    elif best_r2 > 0.20:
        print("⚠️  Moderate results. Consider hyperparameter tuning or ensemble.")
    else:
        print("⚠️  Results need improvement. Review data quality and preprocessing.")
    
    print("\nNext steps:")
    print("1. If ML is better: Use ML model for final predictions")
    print("2. Hyperparameter tuning: Can improve ML models further")
    print("3. Ensemble: Combine CNN + ML predictions for best results")
    print("4. Feature engineering: Add hand-crafted features for ML")


if __name__ == "__main__":
    main()

