# BMD Prediction from MRI T1 FLAIR - Code Explanation

## Project Overview

This project implements a **deep learning regression model** to predict **Bone Mineral Density (BMD)** from **MRI T1 FLAIR sagittal images**. BMD is a critical medical metric for assessing bone health and osteoporosis risk.

## Architecture Overview

### 1. **Data Pipeline** (`src/DataLoader.py`)

#### Key Concepts:

**Patient-Level Splitting**: 
- Data is split at the **patient level** (not scan level) to prevent data leakage
- If a patient has multiple scans, all scans stay in the same split
- Split ratio: 80% train, 10% validation, 10% test

**Multi-Scan Handling**:
- Some patients have multiple MRI scans (files)
- **Training**: Randomly selects one scan per patient per epoch (prevents label duplication bias)
- **Validation/Test**: Uses all scans per patient, then averages predictions per patient

**2.5D Approach**:
- Instead of using full 3D volumes, the model uses **N adjacent slices** (default: 3) as separate channels
- Input shape: `[Batch, N_SLICES, 256, 256]` (e.g., `[8, 3, 256, 256]`)
- Slices are selected around the middle slice of the sagittal axis

#### Preprocessing Pipeline:

1. **Load NIfTI file** (`.nii.gz` format)
2. **Auto-detect sagittal axis** from image orientation metadata
3. **Select N slices** around the middle:
   - Training: Optional slice jitter (±1 slice) for data augmentation
   - Validation/Test: Deterministic middle slice selection
4. **Per-slice processing**:
   - **Percentile clipping** (0.5-99.5%) to remove outliers
   - **Center crop** (384×384) to focus on spine region
   - **Resize** to 256×256
   - **Normalization**: Z-score or robust normalization (median/IQR)
5. **Stack slices** as channels: `[N, 256, 256]`
6. **Target normalization**: BMD values normalized using train set mean/std

#### Dataset Classes:

- **`PatientOneSampleDataset`**: For training - returns 1 sample per patient (random scan)
- **`MultiScanDataset`**: For validation/test - returns all scans per patient

---

### 2. **Model Architecture** (`src/Model.py`)

#### SimpleCNN:

```
Input: [B, in_ch, 256, 256]  (in_ch = N_SLICES, typically 3)
  ↓
Features (4 blocks):
  - Block 1: 32 channels  → 128×128
  - Block 2: 64 channels  → 64×64
  - Block 3: 128 channels → 32×32
  - Block 4: 256 channels → 16×16
  ↓
AdaptiveAvgPool2d(1) → [B, 256, 1, 1]
  ↓
Regressor:
  - Flatten → [B, 256]
  - Linear(256 → 256) + ReLU + Dropout(0.15)
  - Linear(256 → 64) + ReLU + Dropout(0.105)
  - Linear(64 → 1)
  ↓
Output: [B, 1]  (normalized BMD prediction)
```

**Block Structure**:
- Two 3×3 convolutions with BatchNorm and ReLU
- MaxPool2d(2) for downsampling

**Total Parameters**: ~34 million

---

### 3. **Training Script** (`train.py`)

#### Training Loop Flow:

1. **Initialization**:
   - Set random seeds for reproducibility
   - Detect device (CUDA/MPS/CPU)
   - Enable mixed precision (AMP) if CUDA available

2. **Data Loading**:
   - Create train/val/test loaders with patient-level splits
   - Compute target normalization stats from training set only

3. **Model Setup**:
   - Initialize SimpleCNN with `in_ch = N_SLICES`
   - Loss: `SmoothL1Loss` (Huber loss, beta=0.5)
   - Optimizer: `AdamW` (lr=2e-4, weight_decay=1e-4)
   - Scheduler: `ReduceLROnPlateau` (patience=6, factor=0.5)

4. **Training Epoch**:
   - Forward pass with mixed precision
   - Backward pass with gradient clipping (max_norm=5.0)
   - Update weights

5. **Validation**:
   - Collect predictions for all scans
   - **Aggregate per patient** (average predictions across scans)
   - Denormalize to real BMD scale
   - Compute patient-level metrics: MAE, RMSE, R²
   - Save best model based on validation MAE

6. **Testing**:
   - Load best checkpoint
   - Evaluate on test set with per-patient aggregation
   - Print detailed metrics and prediction examples

#### Key Functions:

- **`train_one_epoch()`**: Single training epoch with AMP support
- **`validate_per_patient()`**: Validation with per-patient aggregation
- **`evaluate_per_patient()`**: Test evaluation with detailed output
- **`_aggregate_per_patient()`**: Averages predictions across scans for each patient

---

### 4. **Configuration** (`src/config.py`)

#### Key Parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_SLICES` | 3 | Number of adjacent slices used as channels |
| `IMG_SIZE` | 256 | Final image size after resize |
| `BATCH_SIZE` | 8 | Batch size for training |
| `EPOCHS` | 80 | Maximum training epochs |
| `LR` | 2e-4 | Learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `HUBER_BETA` | 0.5 | SmoothL1Loss beta parameter |
| `CENTER_CROP` | 384 | Center crop size before resize |
| `SLICE_JITTER` | 1 | Random slice shift for augmentation |
| `CLIP_LO/HI` | 0.5/99.5 | Percentile clipping range |
| `USE_ROBUST_NORM` | False | Use robust (median/IQR) vs z-score normalization |

---

## Data Flow Example

### Training:
```
Patient 123 (has 3 scans: scan1.nii.gz, scan2.nii.gz, scan3.nii.gz)
  ↓
Epoch 1: Randomly picks scan2.nii.gz
  → Extract 3 slices → [3, 256, 256]
  → Normalize BMD label
  → Return: (image, normalized_BMD, patient_id)

Epoch 2: Randomly picks scan1.nii.gz
  → Same process...
```

### Validation/Test:
```
Patient 123 (all scans used)
  ↓
Process scan1.nii.gz → prediction₁
Process scan2.nii.gz → prediction₂
Process scan3.nii.gz → prediction₃
  ↓
Average: (prediction₁ + prediction₂ + prediction₃) / 3
  ↓
Compare with true BMD label
```

---

## Evaluation Metrics

All metrics are computed **per-patient** (after aggregating scan-level predictions):

1. **MAE** (Mean Absolute Error): Average absolute difference
2. **RMSE** (Root Mean Squared Error): Square root of mean squared error
3. **R²** (Coefficient of Determination): How well model explains variance
   - R² = 1.0: Perfect predictions
   - R² = 0.0: Model performs as well as predicting the mean
   - R² < 0.0: Model performs worse than predicting the mean

---

## Important Design Decisions

### 1. **Patient-Level Evaluation**
- Medical data often has multiple scans per patient
- Averaging predictions per patient provides more stable and clinically relevant metrics
- Prevents overfitting to scan-level variations

### 2. **2.5D Approach**
- Full 3D CNNs are memory-intensive
- Using adjacent slices as channels captures spatial context efficiently
- Configurable `N_SLICES` allows experimentation

### 3. **Robust Preprocessing**
- Percentile clipping removes MRI artifacts and outliers
- Center crop focuses on spine region
- Robust normalization (optional) handles non-Gaussian intensity distributions

### 4. **Training Strategy**
- Random scan selection per patient prevents label duplication
- Slice jitter provides data augmentation
- Gradient clipping prevents exploding gradients
- Mixed precision training speeds up training on modern GPUs

---

## File Structure

```
BMD_T1_Flair-main/
├── train.py              # Main training script
├── best_model.pth        # Saved best model checkpoint
├── data/
│   ├── metadata.xlsx     # Patient ID → BMD mapping
│   └── Sagittal_T1_FLAIR/  # NIfTI files (.nii.gz)
├── src/
│   ├── config.py        # Configuration parameters
│   ├── DataLoader.py    # Data loading and preprocessing
│   ├── Model.py         # SimpleCNN architecture
│   └── model_reg.py     # Alternative models (not used)
└── docs/
    └── CODE_EXPLANATION.md  # This file
```

---

## Usage

```bash
# Train the model
python train.py
```

The script will:
1. Load and preprocess data
2. Train for specified epochs
3. Save best model based on validation MAE
4. Evaluate on test set and print metrics

---

## Current Status & Improvements

Based on the README, the model achieved:
- **Best R²**: 0.32 (previous version)
- **Current focus**: Patient-level evaluation with improved preprocessing

### Potential Improvements:
- Transfer learning with pretrained models
- More sophisticated architectures (see `model_reg.py` for alternatives)
- Advanced data augmentation
- Cross-validation for better generalization
- Hyperparameter tuning

---

## Key Takeaways

1. **Medical imaging requires careful data handling**: Patient-level splits prevent data leakage
2. **Multi-scan aggregation**: Averaging predictions across scans improves robustness
3. **Preprocessing is critical**: Center crop, percentile clipping, and normalization significantly impact performance
4. **2.5D approach**: Efficient alternative to full 3D processing
5. **Evaluation at the right level**: Patient-level metrics are more clinically meaningful than scan-level metrics

