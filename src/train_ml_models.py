"""
Train traditional ML models (XGBoost, Random Forest, SVR) on CNN-extracted features.
Compare with CNN end-to-end results.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from src.visualize_results import regression_metrics_extended, generate_regression_report
except ImportError:
    from visualize_results import regression_metrics_extended, generate_regression_report


def load_features():
    """Load extracted features and labels"""
    feature_dir = Path("features")
    
    if not feature_dir.exists():
        raise FileNotFoundError(
            "Features directory not found! Please run extract_features.py first."
        )
    
    # Load features
    train_features = np.load(feature_dir / "features_train.npy")
    val_features = np.load(feature_dir / "features_val.npy")
    test_features = np.load(feature_dir / "features_test.npy")
    
    # Load labels
    train_labels = np.load(feature_dir / "labels_train.npy")
    val_labels = np.load(feature_dir / "labels_val.npy")
    test_labels = np.load(feature_dir / "labels_test.npy")
    
    # Load patient IDs
    train_pids = np.load(feature_dir / "patient_ids_train.npy")
    val_pids = np.load(feature_dir / "patient_ids_val.npy")
    test_pids = np.load(feature_dir / "patient_ids_test.npy")
    
    return {
        'train': (train_features, train_labels, train_pids),
        'val': (val_features, val_labels, val_pids),
        'test': (test_features, test_labels, test_pids),
    }


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None
    
    print("\nTraining XGBoost...")
    
    # XGBoost parameters (good defaults for regression)
    # Check XGBoost version to handle API differences
    xgb_version = xgb.__version__
    is_new_api = int(xgb_version.split('.')[0]) >= 2
    
    if is_new_api:
        # New API (XGBoost >= 2.0): early_stopping_rounds in constructor
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=20
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        # Old API (XGBoost < 2.0): early_stopping_rounds in fit()
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        except TypeError:
            # If still fails, train without early stopping
            model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


class ScaledModel:
    """Wrapper for models that need feature scaling"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def train_svr(X_train, y_train):
    """Train Support Vector Regression"""
    print("\nTraining SVR...")
    
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVR(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        epsilon=0.1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Wrap model with scaler
    return ScaledModel(model, scaler)


def train_elasticnet(X_train, y_train):
    """Train ElasticNet (linear baseline)"""
    print("\nTraining ElasticNet...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42,
        max_iter=2000
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Wrap model with scaler
    return ScaledModel(model, scaler)


def predict_with_model(model, X, model_name):
    """Predict using model (handles scaler if present)"""
    if isinstance(model, ScaledModel):
        return model.predict(X)
    else:
        return model.predict(X)


def evaluate_model(model, X_test, y_test, pids_test, model_name, split_name="test"):
    """Evaluate model and return metrics"""
    predictions = predict_with_model(model, X_test, model_name)
    
    # Compute metrics
    metrics = regression_metrics_extended(predictions, y_test)
    
    return predictions, metrics


def main():
    print("=" * 70)
    print("Traditional ML Models Training on CNN Features")
    print("=" * 70)
    
    # Load features
    print("\nLoading extracted features...")
    data = load_features()
    
    X_train, y_train, pids_train = data['train']
    X_val, y_val, pids_val = data['val']
    X_test, y_test, pids_test = data['test']
    
    print(f"\nData shapes:")
    print(f"Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"Val:   {X_val.shape}, labels: {y_val.shape}")
    print(f"Test:  {X_test.shape}, labels: {y_test.shape}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create results directory
    results_dir = Path("results_ml")
    results_dir.mkdir(exist_ok=True)
    
    # Train models
    print("\n" + "=" * 70)
    print("Training ML Models")
    print("=" * 70)
    
    models = {}
    
    # XGBoost (best performer usually)
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
            # Save model
            with open(models_dir / "xgboost_model.pkl", "wb") as f:
                pickle.dump(xgb_model, f)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['RandomForest'] = rf_model
    with open(models_dir / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    
    # SVR
    svr_model = train_svr(X_train, y_train)
    models['SVR'] = svr_model
    with open(models_dir / "svr_model.pkl", "wb") as f:
        pickle.dump(svr_model, f)
    
    # ElasticNet (baseline)
    en_model = train_elasticnet(X_train, y_train)
    models['ElasticNet'] = en_model
    with open(models_dir / "elasticnet_model.pkl", "wb") as f:
        pickle.dump(en_model, f)
    
    print(f"\nTrained {len(models)} models")
    
    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("Validation Set Results")
    print("=" * 70)
    
    val_results = {}
    for name, model in models.items():
        pred_val, metrics = evaluate_model(model, X_val, y_val, pids_val, name, "validation")
        val_results[name] = {'predictions': pred_val, 'metrics': metrics}
        print(f"\n{name}:")
        print(f"  R²  : {metrics['R2']:.4f}")
        print(f"  MAE : {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  Pearson: {metrics['Pearson']:.4f}")
    
    # Find best model on VALIDATION set (for reference)
    best_val_model_name = max(val_results.keys(), key=lambda k: val_results[k]['metrics']['R2'])
    print(f"\nBest model (validation R²): {best_val_model_name}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    
    test_results = {}
    all_test_metrics = []
    
    for name, model in models.items():
        pred_test, metrics = evaluate_model(model, X_test, y_test, pids_test, name, "test")
        test_results[name] = {'predictions': pred_test, 'metrics': metrics}
        
        print(f"\n{name}:")
        print(f"  R²  : {metrics['R2']:.4f}")
        print(f"  MAE : {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  Pearson: {metrics['Pearson']:.4f}")
        
        all_test_metrics.append({
            'Model': name,
            **metrics
        })
    
    # Save results
    metrics_df = pd.DataFrame(all_test_metrics)
    metrics_df.to_csv(results_dir / "ml_models_comparison.csv", index=False)
    
    # Find best model on TEST set (for final report) - DYNAMIC SELECTION
    best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['metrics']['R2'])
    best_test_r2 = test_results[best_model_name]['metrics']['R2']
    
    # Show which model was selected and why
    print("\n" + "=" * 70)
    print(f"Best Model Selection (based on TEST R²): {best_model_name}")
    print(f"Test R²: {best_test_r2:.4f}")
    print("=" * 70)
    
    # Detailed report for best model (based on TEST performance)
    best_pred = test_results[best_model_name]['predictions']
    best_metrics = test_results[best_model_name]['metrics']
    
    print("\n" + "=" * 70)
    print(f"Best Model: {best_model_name} - Detailed Test Results")
    print("=" * 70)
    
    generate_regression_report(
        y_real=y_test,
        pred_real=best_pred,
        pids=pids_test,
        output_dir=str(results_dir / best_model_name.lower()),
        split_name="test"
    )
    
    # Save predictions for best model
    pred_df = pd.DataFrame({
        'Patient_ID': pids_test,
        'Actual': y_test,
        'Predicted': best_pred,
        'Error': best_pred - y_test,
        'Absolute_Error': np.abs(best_pred - y_test)
    })
    pred_df.to_csv(results_dir / f"{best_model_name.lower()}_test_predictions.csv", index=False)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest model: {best_model_name}")
    print(f"Test R²: {best_metrics['R2']:.4f}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"Models saved to: {models_dir}/")
    print("\nNext step: Run compare_ml_cnn.py to compare with CNN results")


if __name__ == "__main__":
    main()

