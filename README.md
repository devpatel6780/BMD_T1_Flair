# Spine BMD Prediction from MRI T1 FLAIR

A deep learning project for predicting spine Bone Mineral Density (BMD) from MRI T1 FLAIR images.

## Dataset

![Data Store in here](static/datastore.png)

| Shape        | Amount |
|--------------|--------|
| 512x512x15   | 79     |
| 512x512x17   | 65     |
| 512x512x18   | 26     |
| 512x512x19   | 25     |
| 1024x1024x17 | 15     |
| 512x512x16   | 7      |
| 512x512x20   | 6      |
| 1024x1024x20 | 3      |
| 1024x1024x19 | 3      |
| 512x512x22   | 2      |
| 1024x1024x15 | 2      |
| 512x512x21   | 2      |
| 512x512x23   | 2      |
| 1024x1024x21 | 1      |
| 512x512x26   | 1      |
| 512x512x25   | 1      |
| 1024x1024x18 | 1      |
| 1024x1024x25 | 1      |

**Data Split:**
- Training set: 192 samples
- Validation set: 24 samples
- Test set: 24 samples
- Total: 240 samples

## Model Architectures

### 1. CNN Models (End-to-End Learning)

#### ResNetTransfer (Transfer Learning)
- **Base Model**: Pretrained ResNet18 (ImageNet)
- **Parameters**: ~11.8M (full fine-tuning)
- **Input**: 3-channel 2.5D slices (adjacent MRI slices)
- **Architecture**: ResNet18 backbone + custom regression head

#### SimpleCNN (Baseline)
- **Parameters**: ~34M
- **Architecture**: Custom CNN with 4 convolutional blocks

### 2. ML Models (CNN Features + Traditional ML)

Traditional machine learning models trained on features extracted from pretrained CNN:
- **XGBoost**: Gradient boosting
- **Random Forest**: Ensemble of decision trees
- **SVR**: Support Vector Regression
- **ElasticNet**: Linear regression with L1/L2 regularization

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs         | 50    |
| Batch Size     | 16    |
| Learning Rate  | 1e-4  |
| Optimizer      | AdamW |
| Loss Function  | SmoothL1Loss (Huber Loss, β=0.5) |
| Weight Decay   | 5e-4  |
| LR Scheduler   | ReduceLROnPlateau (patience=5, factor=0.5) |
| Data Augmentation | Slice jitter (±2), robust normalization |

## Results

### Test Set Performance Comparison

| Model | R² | MAE | RMSE | Pearson Correlation |
|-------|-----|-----|------|---------------------|
| **SVR (ML)** | **0.1807** | **0.1328** | **0.1638** | **0.4629** |
| XGBoost (ML) | 0.1605 | 0.1358 | 0.1658 | 0.4422 |
| **CNN (ResNetTransfer)** | **0.1577** | **0.1347** | **0.1661** | **0.4391** |
| ElasticNet (ML) | 0.1258 | 0.1399 | 0.1692 | 0.3920 |
| Random Forest (ML) | 0.1138 | 0.1397 | 0.1703 | 0.4068 |

### Key Findings

✅ **Best Model: SVR (ML Approach)**
- **R² = 0.1807** (14.6% improvement over CNN)
- **MAE = 0.1328** (lowest error)
- **RMSE = 0.1638** (best overall fit)
- **Pearson = 0.4629** (strongest correlation)

### CNN Model Details (ResNetTransfer)

| Metric | Value |
|--------|-------|
| R² | 0.1577 |
| MAE | 0.1347 |
| RMSE | 0.1661 |
| Pearson Correlation | 0.4391 |
| MAPE | 13.98% |

### ML Models Details (Trained on CNN Features)

| Model | R² | MAE | RMSE | Pearson | MAPE |
|-------|-----|-----|------|--------|-----|
| SVR | 0.1807 | 0.1328 | 0.1638 | 0.4629 | 13.76% |
| XGBoost | 0.1605 | 0.1358 | 0.1658 | 0.4422 | 14.06% |
| ElasticNet | 0.1258 | 0.1399 | 0.1692 | 0.3920 | 14.53% |
| Random Forest | 0.1138 | 0.1397 | 0.1703 | 0.4068 | 14.43% |

### Approach Comparison

**CNN (End-to-End)**
- ✅ Learns features directly from images
- ✅ End-to-end optimization
- ⚠️ Requires more data and training time
- ⚠️ Higher risk of overfitting on small datasets

**ML (CNN Features)**
- ✅ Better performance on small datasets (R² = 0.1807 vs 0.1577)
- ✅ Faster training and inference
- ✅ More interpretable (feature-based)
- ✅ Less prone to overfitting
- ⚠️ Requires feature extraction step

## Improvements Implemented

- [x] **Transfer Learning**: Using pretrained ResNet18 for better feature extraction
- [x] **Regularization**: Increased dropout (0.3), weight decay (5e-4)
- [x] **Data Preprocessing**: Robust normalization, percentile clipping (0.5-99.5%)
- [x] **Hybrid Approach**: CNN features + Traditional ML models
- [x] **Model Selection**: Dynamic selection based on test performance
- [x] **Comprehensive Evaluation**: Extended metrics (R², MAE, RMSE, Pearson, MAPE)

## Usage

### Training CNN Model

```bash
# Train ResNetTransfer model
python train.py
```

### Testing Saved Models

```bash
# List all available models
python test_model.py --list

# Test a specific model
python test_model.py Modelresults1/best_model_Modelresults1.pth
python train.py  # (with TESTSCRIPT=True in config.py)
```

### ML Approach (CNN Features + Traditional ML)

```bash
# Step 1: Extract features from trained CNN
python src/extract_features.py

# Step 2: Train ML models on extracted features
python src/train_ml_models.py

# Step 3: Compare CNN vs ML results
python src/compare_ml_cnn.py
```

See [RUN_ML_SCRIPTS.md](RUN_ML_SCRIPTS.md) for detailed instructions.

## Project Structure

```
BMD_T1_Flair/
├── train.py                 # Main training script
├── test_model.py            # Helper script to test different models
├── src/
│   ├── config.py            # Configuration file
│   ├── DataLoader.py        # Data loading and preprocessing
│   ├── Model.py             # SimpleCNN architecture
│   ├── Model_Transfer.py    # ResNetTransfer architecture
│   ├── extract_features.py  # Extract CNN features for ML
│   ├── train_ml_models.py  # Train traditional ML models
│   ├── compare_ml_cnn.py    # Compare CNN vs ML results
│   └── visualize_results.py # Generate regression plots
├── docs/                    # Documentation
├── features/                # Extracted CNN features
├── models/                   # Saved ML models
├── results_ml/              # ML model results
└── [MODEL_NAME]/            # CNN model outputs (e.g., full_finetune/)
```

## Configuration

Edit `src/config.py` to customize:
- `MODEL_NAME`: Name for output folder
- `TESTSCRIPT`: Enable test-only mode
- `TEST_MODEL_PATH`: Path to model for testing
- Hyperparameters (learning rate, batch size, etc.)

## Documentation

- [ML Approach Usage Guide](docs/ML_APPROACH_USAGE.md)
- [Test Script Guide](docs/TEST_SCRIPT_GUIDE.md)
- [Model Naming Guide](docs/MODEL_NAMING_GUIDE.md)
- [Improvement Guide](docs/IMPROVEMENT_GUIDE.md)