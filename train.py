# ########################################
# # best r2 0.32
# ########################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random

# from src.DataLoader import create_dataloaders
# from src.Model import SimpleCNN


# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # Determinism (slower but stable)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def train():
#     set_seed(42)

#     # Device setup
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         print("Using CUDA GPU")
#     elif torch.backends.mps.is_available():
#         device = torch.device('mps')
#         print("Using Apple MPS")
#     else:
#         device = torch.device('cpu')
#         print("Using CPU")

#     # Hyperparameters
#     epochs = 80
#     batch_size = 16
#     learning_rate = 2e-4
#     weight_decay = 1e-4

#     # Data paths
#     data_dir = 'data/Sagittal_T1_FLAIR'
#     xlsx_path = 'data/metadata.xlsx'

#     print("\nLoading data...")
#     train_loader, val_loader, test_loader, (y_mean, y_std) = create_dataloaders(
#         data_dir=data_dir,
#         xlsx_path=xlsx_path,
#         batch_size=batch_size,
#         img_size=256,
#         random_state=42,
#         num_workers=0
#     )

#     # Model
#     model = SimpleCNN(in_ch=3).to(device)
#     print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

#     # Loss: Huber (Smooth L1)
#     criterion = nn.SmoothL1Loss(beta=0.8)

#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Scheduler: reduce LR when val stalls
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6
#     )

#     best_val_loss = float('inf')
#     best_path = 'best_model.pth'

#     print("\nStart training...")
#     print("=" * 70)

#     for epoch in range(epochs):
#         # ---- Training ----
#         model.train()
#         train_loss = 0.0

#         for images, labels in train_loader:
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             outputs = model(images).squeeze(1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             train_loss += loss.item()

#         avg_train_loss = train_loss / max(1, len(train_loader))

#         # ---- Validation ----
#         model.eval()
#         val_loss = 0.0

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images = images.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True)

#                 outputs = model(images).squeeze(1)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / max(1, len(val_loader))
#         scheduler.step(avg_val_loss)

#         # Save best checkpoint (includes target stats)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save({
#                 "state_dict": model.state_dict(),
#                 "y_mean": float(y_mean),
#                 "y_std": float(y_std),
#             }, best_path)
#             save_marker = " *"
#         else:
#             save_marker = ""

#         lr_now = optimizer.param_groups[0]["lr"]
#         print(f"Epoch [{epoch+1:3d}/{epochs}] "
#               f"LR: {lr_now:.2e} | "
#               f"Train Loss: {avg_train_loss:.4f} | "
#               f"Val Loss: {avg_val_loss:.4f}{save_marker}")

#     print("=" * 70)
#     print(f"Training complete! Best val loss: {best_val_loss:.4f}")

#     # Test
#     print("\nLoading best model for testing...")
#     ckpt = torch.load(best_path, map_location=device)
#     model.load_state_dict(ckpt["state_dict"])
#     y_mean = float(ckpt["y_mean"])
#     y_std = float(ckpt["y_std"])

#     evaluate(model, test_loader, device, y_mean, y_std)


# def evaluate(model, test_loader, device, y_mean, y_std):
#     model.eval()
#     preds_norm = []
#     y_norm = []

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device, non_blocking=True)
#             outputs = model(images).squeeze(1).detach().cpu().numpy()
#             preds_norm.extend(outputs.tolist())
#             y_norm.extend(labels.numpy().tolist())

#     preds_norm = np.array(preds_norm, dtype=np.float32)
#     y_norm = np.array(y_norm, dtype=np.float32)

#     # ---- Denormalize back to real BMD scale ----
#     predictions = preds_norm * y_std + y_mean
#     actuals = y_norm * y_std + y_mean

#     mse = float(np.mean((predictions - actuals) ** 2))
#     rmse = float(np.sqrt(mse))
#     mae = float(np.mean(np.abs(predictions - actuals)))

#     ss_res = float(np.sum((actuals - predictions) ** 2))
#     ss_tot = float(np.sum((actuals - float(np.mean(actuals))) ** 2))
#     r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

#     print("\n" + "=" * 44)
#     print("              Test Results")
#     print("=" * 44)
#     print(f"MSE  (Mean Squared Error):          {mse:.4f}")
#     print(f"RMSE (Root Mean Squared Error):     {rmse:.4f}")
#     print(f"MAE  (Mean Absolute Error):         {mae:.4f}")
#     print(f"R2   (Coefficient of Determination): {r2:.4f}")
#     print("=" * 44)

#     # Show examples
#     print("\nPrediction examples (first 10):")
#     print("-" * 34)
#     print(f"{'Actual':^12} | {'Predicted':^12}")
#     print("-" * 34)
#     for i in range(min(10, len(actuals))):
#         print(f"{actuals[i]:^12.3f} | {predictions[i]:^12.3f}")
#     print("-" * 34)


# if __name__ == '__main__':
#     train()



# train.py

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    N_SLICES, IMG_SIZE, BATCH_SIZE, RANDOM_STATE, NUM_WORKERS,
    EPOCHS, LR, WEIGHT_DECAY, HUBER_BETA,
    DATA_DIR, XLSX_PATH, BEST_PATH, RESULTS_DIR, OUTPUT_DIR,
    CENTER_CROP, SLICE_JITTER, CLIP_LO, CLIP_HI, USE_ROBUST_NORM,
    TESTSCRIPT, MODEL_NAME, TEST_MODEL_PATH
)
from src.DataLoader import create_dataloaders
from src.Model_Transfer import ResNetTransfer
from src.Model import SimpleCNN
from src.visualize_results import generate_regression_report


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def regression_metrics(pred_real: np.ndarray, y_real: np.ndarray):
    pred_real = pred_real.astype(np.float32)
    y_real = y_real.astype(np.float32)

    mse = float(np.mean((pred_real - y_real) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_real - y_real)))

    ss_res = float(np.sum((y_real - pred_real) ** 2))
    ss_tot = float(np.sum((y_real - float(np.mean(y_real))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    return mse, rmse, mae, r2


def _aggregate_per_patient(preds_norm: np.ndarray, y_norm: np.ndarray, pids: np.ndarray):
    """
    Average predictions across scans for each patient.
    Labels should be constant per patient; we take the first label.
    Returns arrays aligned per patient: (pred_norm_patient, y_norm_patient)
    """
    pids = pids.astype(np.int64)
    preds_norm = preds_norm.astype(np.float32)
    y_norm = y_norm.astype(np.float32)

    uniq = np.unique(pids)
    pred_p, y_p = [], []

    for pid in uniq:
        m = (pids == pid)
        pred_p.append(float(np.mean(preds_norm[m])))
        # label should be identical for all scans of the patient; take first
        y_p.append(float(y_norm[m][0]))

    return np.array(pred_p, dtype=np.float32), np.array(y_p, dtype=np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    running = 0.0

    for batch in loader:
        images, labels, _pids = batch  # train loader yields (x, y, pid)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images).squeeze(1)  # [B]
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        running += float(loss.item())

    return running / max(1, len(loader))


@torch.no_grad()
def validate_per_patient(model, loader, criterion, device, y_mean, y_std, return_predictions=False):
    """
    VAL loader is multi-scan (scan-level batches), but we report TRUE patient-level metrics
    by averaging predictions per patient.
    
    Args:
        return_predictions: If True, also return (pred_real, y_real, unique_pids)
    """
    model.eval()

    scan_loss = 0.0
    preds_norm, y_norm, pids = [], [], []

    for images, labels, pid in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        scan_loss += float(loss.item())

        preds_norm.append(outputs.detach().cpu().numpy())
        y_norm.append(labels.detach().cpu().numpy())
        pids.append(pid.detach().cpu().numpy())

    scan_loss = scan_loss / max(1, len(loader))

    preds_norm = np.concatenate(preds_norm, axis=0)
    y_norm = np.concatenate(y_norm, axis=0)
    pids = np.concatenate(pids, axis=0)

    pred_p_norm, y_p_norm = _aggregate_per_patient(preds_norm, y_norm, pids)

    pred_real = pred_p_norm * float(y_std) + float(y_mean)
    y_real = y_p_norm * float(y_std) + float(y_mean)

    mse, rmse, mae, r2 = regression_metrics(pred_real, y_real)
    
    unique_pids = np.unique(pids)
    
    if return_predictions:
        return scan_loss, mse, rmse, mae, r2, len(unique_pids), (pred_real, y_real, unique_pids)
    return scan_loss, mse, rmse, mae, r2, len(unique_pids)


@torch.no_grad()
def evaluate_per_patient(model, loader, device, y_mean, y_std, output_dir="results", split_name="test"):
    """
    TEST loader is multi-scan; we average per patient.
    Generates comprehensive regression report with visualizations.
    """
    model.eval()

    preds_norm, y_norm, pids = [], [], []

    for images, labels, pid in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images).squeeze(1).detach().cpu().numpy()
        preds_norm.append(outputs.astype(np.float32))
        y_norm.append(labels.detach().cpu().numpy().astype(np.float32))
        pids.append(pid.detach().cpu().numpy().astype(np.int64))

    preds_norm = np.concatenate(preds_norm, axis=0)
    y_norm = np.concatenate(y_norm, axis=0)
    pids = np.concatenate(pids, axis=0)

    pred_p_norm, y_p_norm = _aggregate_per_patient(preds_norm, y_norm, pids)
    unique_pids = np.unique(pids)

    pred_real = pred_p_norm * float(y_std) + float(y_mean)
    y_real = y_p_norm * float(y_std) + float(y_mean)

    # Generate comprehensive report with visualizations
    metrics = generate_regression_report(
        y_real=y_real,
        pred_real=pred_real,
        pids=unique_pids,
        output_dir=output_dir,
        split_name=split_name
    )

    return metrics


@torch.no_grad()
def test_only_evaluation(model, loader, device, y_mean, y_std, split_name="test"):
    """
    Test-only evaluation: Shows results/metrics only, no plots or CSV files.
    Used when TESTSCRIPT = True in config.
    """
    model.eval()

    preds_norm, y_norm, pids = [], [], []

    for images, labels, pid in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images).squeeze(1).detach().cpu().numpy()
        preds_norm.append(outputs.astype(np.float32))
        y_norm.append(labels.detach().cpu().numpy().astype(np.float32))
        pids.append(pid.detach().cpu().numpy().astype(np.int64))

    preds_norm = np.concatenate(preds_norm, axis=0)
    y_norm = np.concatenate(y_norm, axis=0)
    pids = np.concatenate(pids, axis=0)

    pred_p_norm, y_p_norm = _aggregate_per_patient(preds_norm, y_norm, pids)
    unique_pids = np.unique(pids)

    pred_real = pred_p_norm * float(y_std) + float(y_mean)
    y_real = y_p_norm * float(y_std) + float(y_mean)

    # Compute metrics
    mse, rmse, mae, r2 = regression_metrics(pred_real, y_real)
    
    # Extended metrics
    from src.visualize_results import regression_metrics_extended
    metrics = regression_metrics_extended(pred_real, y_real)
    
    # Print results only (no plots)
    print("\n" + "=" * 70)
    print(f"         {split_name.upper()} Results (Per-Patient) - Metrics Only")
    print("=" * 70)
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
    print("=" * 70)
    
    # Show prediction examples
    print(f"\nPrediction Examples (first 10 patients):")
    print("-" * 50)
    print(f"{'Patient ID':^12} | {'Actual':^12} | {'Predicted':^12} | {'Error':^12}")
    print("-" * 50)
    for i in range(min(10, len(y_real))):
        error = pred_real[i] - y_real[i]
        pid_str = str(unique_pids[i]) if i < len(unique_pids) else "N/A"
        print(f"{pid_str:^12} | {y_real[i]:^12.3f} | {pred_real[i]:^12.3f} | {error:^12.3f}")
    print("-" * 50)
    
    return metrics


def train():
    set_seed(RANDOM_STATE)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
        use_amp = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
        use_amp = False
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_amp = False

    print("\nLoading data...")
    train_loader, val_loader, test_loader, (y_mean, y_std) = create_dataloaders(
        data_dir=DATA_DIR,
        xlsx_path=XLSX_PATH,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        random_state=RANDOM_STATE,
        num_workers=NUM_WORKERS,
        n_slices=N_SLICES,
        center_crop=CENTER_CROP,
        clip_lo=CLIP_LO,
        clip_hi=CLIP_HI,
        use_robust_norm=USE_ROBUST_NORM,
        slice_jitter=SLICE_JITTER,
    )

    # Sanity: model input channels must match dataset channels
    first_images, _, _ = next(iter(train_loader))
    in_ch = int(first_images.shape[1])
    if in_ch != int(N_SLICES):
        raise RuntimeError(f"Channel mismatch: loader gives {in_ch} but config N_SLICES={N_SLICES}")

    print(f"\nUsing N_SLICES={N_SLICES} => model in_ch={in_ch}")

    # ===== TEST-ONLY MODE =====
    if TESTSCRIPT:
        print("\n" + "=" * 70)
        print("TEST-ONLY MODE: Loading saved model and evaluating")
        print("=" * 70)
        print(f"Model name: {MODEL_NAME}")
        print(f"Loading model from: {BEST_PATH}")
        
        # Check if model file exists
        if not os.path.exists(BEST_PATH):
            raise FileNotFoundError(
                f"Model file not found: {BEST_PATH}\n"
                f"Make sure MODEL_NAME in config.py matches the folder name containing the saved model."
            )
        
        # Load checkpoint
        ckpt = torch.load(BEST_PATH, map_location=device)
        
        # Get model configuration from checkpoint
        in_ch_ckpt = int(ckpt.get("in_ch", in_ch))
        if in_ch_ckpt != in_ch:
            print(f"Warning: Config N_SLICES={N_SLICES} (in_ch={in_ch}) but checkpoint has in_ch={in_ch_ckpt}")
            print(f"Using checkpoint value: in_ch={in_ch_ckpt}")
            in_ch = in_ch_ckpt
        
        # Detect model architecture from checkpoint keys
        state_dict_keys = list(ckpt["state_dict"].keys())
        is_simple_cnn = any("features.0.0.weight" in key for key in state_dict_keys)
        
        if is_simple_cnn:
            print("Detected model architecture: SimpleCNN")
            model = SimpleCNN(in_ch=in_ch, dropout=0.35).to(device)
        else:
            print("Detected model architecture: ResNetTransfer")
            model = ResNetTransfer(in_ch=in_ch, dropout=0.3).to(device)
        
        model.load_state_dict(ckpt["state_dict"])
        
        # Get normalization stats from checkpoint
        y_mean = float(ckpt["y_mean"])
        y_std = float(ckpt["y_std"])
        
        print(f"Model loaded successfully!")
        print(f"Normalization stats: mean={y_mean:.6f}, std={y_std:.6f}")
        print("=" * 70)
        
        # Evaluate on test set (results only, no plots)
        print("\nEvaluating on TEST set...")
        test_only_evaluation(model, test_loader, device, y_mean, y_std, split_name="test")
        
        # Also evaluate on validation set
        print("\nEvaluating on VALIDATION set...")
        test_only_evaluation(model, val_loader, device, y_mean, y_std, split_name="validation")
        
        print("\n" + "=" * 70)
        print("Test evaluation complete!")
        print("=" * 70)
        return

    # ===== TRAINING MODE (Normal) =====
    # Use transfer learning (much better for small datasets)
    model = ResNetTransfer(in_ch=in_ch, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (full fine-tuning: all layers trainable)")

    criterion = nn.SmoothL1Loss(beta=HUBER_BETA)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_mae = float("inf")
    
    # Track metrics for learning curves
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []

    print("\nStart training...")
    print("=" * 106)
    print("Epoch      |    LR    | TrainLoss | ValScanLoss | ValMAE(P) | ValRMSE(P) |  ValR2(P) | Saved")
    print("-" * 106)

    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)

        val_scan_loss, val_mse, val_rmse, val_mae, val_r2, n_val_pat = validate_per_patient(
            model, val_loader, criterion, device, y_mean, y_std
        )
        
        # Track metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_scan_loss)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        # step on true per-patient MAE
        scheduler.step(val_mae)

        saved = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "y_mean": float(y_mean),
                    "y_std": float(y_std),
                    "in_ch": int(in_ch),
                    "n_slices": int(N_SLICES),
                    "center_crop": int(CENTER_CROP),
                    "clip_lo": float(CLIP_LO),
                    "clip_hi": float(CLIP_HI),
                    "use_robust_norm": bool(USE_ROBUST_NORM),
                },
                BEST_PATH,
            )
            saved = " *"

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch+1:3d}/{EPOCHS:<3d} | {lr_now:8.2e} | {avg_train_loss:9.4f} | {val_scan_loss:10.4f} "
            f"| {val_mae:8.4f} | {val_rmse:9.4f} | {val_r2:9.4f} |{saved}"
        )

    print("=" * 106)
    print(f"Training complete! Best VAL MAE (per-patient, real scale): {best_val_mae:.4f}")

    print("\nLoading best model for testing...")
    ckpt = torch.load(BEST_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])

    # Generate comprehensive test report with visualizations
    print("\n" + "=" * 60)
    print("Generating comprehensive test set evaluation report...")
    print("=" * 60)
    print(f"All outputs saved in folder: {OUTPUT_DIR}/")
    print(f"  - Model: {BEST_PATH}")
    print(f"  - Results: {RESULTS_DIR}/")
    print("=" * 60)
    
    evaluate_per_patient(
        model, test_loader, device, y_mean, y_std,
        output_dir=RESULTS_DIR, split_name="test"
    )
    
    # Also generate validation report with learning curves
    print("\n" + "=" * 60)
    print("Generating validation set report with learning curves...")
    print("=" * 60)
    
    val_scan_loss, val_mse, val_rmse, val_mae, val_r2, n_val_pat, (val_pred, val_actual, val_pids) = validate_per_patient(
        model, val_loader, criterion, device, y_mean, y_std, return_predictions=True
    )
    
    generate_regression_report(
        y_real=val_actual,
        pred_real=val_pred,
        pids=val_pids,
        output_dir=RESULTS_DIR,
        split_name="validation",
        train_losses=train_losses,
        val_losses=val_losses,
        val_metrics={'mae': val_maes, 'r2': val_r2s}
    )


if __name__ == "__main__":
    train()
