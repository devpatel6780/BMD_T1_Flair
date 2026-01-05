"""
Visualization and reporting utilities for regression results.
Generates comprehensive plots and metrics for regression model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import pandas as pd


def regression_metrics_extended(pred_real: np.ndarray, y_real: np.ndarray):
    """
    Compute extended regression metrics.
    
    Returns:
        dict with all metrics
    """
    pred_real = pred_real.astype(np.float32)
    y_real = y_real.astype(np.float32)
    
    # Basic metrics
    mse = float(np.mean((pred_real - y_real) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_real - y_real)))
    
    # R²
    ss_res = float(np.sum((y_real - pred_real) ** 2))
    ss_tot = float(np.sum((y_real - float(np.mean(y_real))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
    
    # Pearson correlation
    vx = pred_real - pred_real.mean()
    vy = y_real - y_real.mean()
    pearson = float((vx * vy).sum() / (np.sqrt((vx * vx).sum() + 1e-8) * np.sqrt((vy * vy).sum() + 1e-8)))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = float(np.mean(np.abs((y_real - pred_real) / (y_real + 1e-8)) * 100))
    
    # Median Absolute Error
    median_ae = float(np.median(np.abs(pred_real - y_real)))
    
    # Error statistics
    errors = pred_real - y_real
    error_std = float(np.std(errors))
    error_min = float(np.min(errors))
    error_max = float(np.max(errors))
    error_q25 = float(np.percentile(errors, 25))
    error_q75 = float(np.percentile(errors, 75))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Median_AE': median_ae,
        'R2': r2,
        'Pearson': pearson,
        'MAPE': mape,
        'Error_Std': error_std,
        'Error_Min': error_min,
        'Error_Max': error_max,
        'Error_Q25': error_q25,
        'Error_Q75': error_q75,
        'Std_Actual': float(np.std(y_real)),
        'Std_Predicted': float(np.std(pred_real)),
        'Mean_Actual': float(np.mean(y_real)),
        'Mean_Predicted': float(np.mean(pred_real)),
    }


def plot_predicted_vs_actual(
    y_real: np.ndarray,
    pred_real: np.ndarray,
    metrics: dict,
    save_path: Optional[str] = None,
    title: str = "Predicted vs Actual"
):
    """
    Create scatter plot of predicted vs actual values with perfect prediction line.
    
    Args:
        y_real: Actual values
        pred_real: Predicted values
        metrics: Dictionary with metrics (R2, MAE, RMSE, Pearson)
        save_path: Path to save figure (optional)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_real, pred_real, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (y=x)
    min_val = min(y_real.min(), pred_real.min())
    max_val = max(y_real.max(), pred_real.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Linear fit line
    z = np.polyfit(y_real, pred_real, 1)
    p = np.poly1d(z)
    ax.plot(y_real, p(y_real), "b--", alpha=0.8, lw=1.5, label=f'Linear Fit (slope={z[0]:.3f})')
    
    ax.set_xlabel('Actual BMD', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted BMD', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add metrics text box
    textstr = f"R² = {metrics['R2']:.4f}\n"
    textstr += f"MAE = {metrics['MAE']:.4f}\n"
    textstr += f"RMSE = {metrics['RMSE']:.4f}\n"
    textstr += f"Pearson = {metrics['Pearson']:.4f}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_residuals(
    y_real: np.ndarray,
    pred_real: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Residual Plot"
):
    """
    Create residual plot (errors vs actual values).
    
    Args:
        y_real: Actual values
        pred_real: Predicted values
        save_path: Path to save figure (optional)
        title: Plot title
    """
    residuals = pred_real - y_real
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Actual
    ax1.scatter(y_real, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Actual BMD', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Actual', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax2.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_error_distribution(
    y_real: np.ndarray,
    pred_real: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Error Distribution"
):
    """
    Create error distribution and comparison plots.
    
    Args:
        y_real: Actual values
        pred_real: Predicted values
        save_path: Path to save figure (optional)
        title: Plot title
    """
    errors = pred_real - y_real
    abs_errors = np.abs(errors)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Absolute error distribution
    ax1.hist(abs_errors, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(abs_errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(abs_errors):.4f}')
    ax1.axvline(np.median(abs_errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(abs_errors):.4f}')
    ax1.legend()
    
    # Actual vs Predicted distributions
    ax2.hist(y_real, bins=15, alpha=0.6, label='Actual', edgecolor='black')
    ax2.hist(pred_real, bins=15, alpha=0.6, label='Predicted', edgecolor='black')
    ax2.set_xlabel('BMD Value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error vs Predicted
    ax3.scatter(pred_real, errors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted BMD', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Error (Predicted - Actual)', fontsize=11, fontweight='bold')
    ax3.set_title('Error vs Predicted', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot for residuals (check normality)
    try:
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax4)
    except ImportError:
        ax4.text(0.5, 0.5, 'Q-Q plot requires scipy', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Q-Q Plot (scipy not available)', fontsize=12, fontweight='bold')
    ax4.set_title('Q-Q Plot (Residual Normality)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_learning_curves(
    train_losses: list,
    val_losses: list,
    val_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_metrics: Optional dict with lists of validation metrics (e.g., {'mae': [...], 'r2': [...]})
        save_path: Path to save figure (optional)
        title: Plot title
    """
    epochs = range(1, len(train_losses) + 1)
    
    if val_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes.flat
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss curves (log scale)
    ax2.semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.semilogy(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Curves (Log Scale)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if val_metrics:
        # Validation MAE
        if 'mae' in val_metrics:
            ax3.plot(epochs, val_metrics['mae'], 'g-', label='Val MAE', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax3.set_ylabel('MAE', fontsize=12, fontweight='bold')
            ax3.set_title('Validation MAE', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Validation R²
        if 'r2' in val_metrics:
            ax4.plot(epochs, val_metrics['r2'], 'm-', label='Val R²', linewidth=2)
            ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax4.set_ylabel('R²', fontsize=12, fontweight='bold')
            ax4.set_title('Validation R²', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def save_predictions_csv(
    y_real: np.ndarray,
    pred_real: np.ndarray,
    pids: Optional[np.ndarray] = None,
    save_path: str = "predictions.csv"
):
    """
    Save predictions to CSV file.
    
    Args:
        y_real: Actual values
        pred_real: Predicted values
        pids: Optional patient IDs
        save_path: Path to save CSV
    """
    data = {
        'Actual': y_real,
        'Predicted': pred_real,
        'Error': pred_real - y_real,
        'Absolute_Error': np.abs(pred_real - y_real),
        'Percentage_Error': np.abs((pred_real - y_real) / (y_real + 1e-8)) * 100
    }
    
    if pids is not None:
        data['Patient_ID'] = pids
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions to: {save_path}")


def generate_regression_report(
    y_real: np.ndarray,
    pred_real: np.ndarray,
    pids: Optional[np.ndarray] = None,
    output_dir: str = "results",
    split_name: str = "test",
    train_losses: Optional[list] = None,
    val_losses: Optional[list] = None,
    val_metrics: Optional[dict] = None
):
    """
    Generate comprehensive regression report with all plots and metrics.
    
    Args:
        y_real: Actual values
        pred_real: Predicted values
        pids: Optional patient IDs
        output_dir: Directory to save all outputs
        split_name: Name of the split (train/val/test)
        train_losses: Optional training losses for learning curves
        val_losses: Optional validation losses for learning curves
        val_metrics: Optional validation metrics for learning curves
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Compute extended metrics
    metrics = regression_metrics_extended(pred_real, y_real)
    
    # Print metrics
    print("\n" + "=" * 60)
    print(f"         Extended Regression Metrics ({split_name.upper()})")
    print("=" * 60)
    print(f"MSE              : {metrics['MSE']:.6f}")
    print(f"RMSE             : {metrics['RMSE']:.6f}")
    print(f"MAE              : {metrics['MAE']:.6f}")
    print(f"Median AE        : {metrics['Median_AE']:.6f}")
    print(f"R²               : {metrics['R2']:.6f}")
    print(f"Pearson Corr     : {metrics['Pearson']:.6f}")
    print(f"MAPE (%)         : {metrics['MAPE']:.4f}")
    print(f"\nError Statistics:")
    print(f"  Std            : {metrics['Error_Std']:.6f}")
    print(f"  Min            : {metrics['Error_Min']:.6f}")
    print(f"  Max            : {metrics['Error_Max']:.6f}")
    print(f"  Q25            : {metrics['Error_Q25']:.6f}")
    print(f"  Q75            : {metrics['Error_Q75']:.6f}")
    print(f"\nDistribution Statistics:")
    print(f"  Actual Mean    : {metrics['Mean_Actual']:.6f}")
    print(f"  Predicted Mean : {metrics['Mean_Predicted']:.6f}")
    print(f"  Actual Std     : {metrics['Std_Actual']:.6f}")
    print(f"  Predicted Std  : {metrics['Std_Predicted']:.6f}")
    print("=" * 60)
    
    # Generate plots
    plot_predicted_vs_actual(
        y_real, pred_real, metrics,
        save_path=str(output_path / f"{split_name}_predicted_vs_actual.png"),
        title=f"Predicted vs Actual - {split_name.upper()}"
    )
    
    plot_residuals(
        y_real, pred_real,
        save_path=str(output_path / f"{split_name}_residuals.png"),
        title=f"Residual Analysis - {split_name.upper()}"
    )
    
    plot_error_distribution(
        y_real, pred_real,
        save_path=str(output_path / f"{split_name}_error_distribution.png"),
        title=f"Error Distribution - {split_name.upper()}"
    )
    
    # Learning curves if provided
    if train_losses and val_losses:
        plot_learning_curves(
            train_losses, val_losses, val_metrics,
            save_path=str(output_path / "learning_curves.png"),
            title="Training and Validation Curves"
        )
    
    # Save predictions CSV
    save_predictions_csv(
        y_real, pred_real, pids,
        save_path=str(output_path / f"{split_name}_predictions.csv")
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / f"{split_name}_metrics.csv", index=False)
    
    print(f"\nAll results saved to: {output_path}")
    return metrics

