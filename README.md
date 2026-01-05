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

## Model Architecture

- **Model**: SimpleCNN
- **Parameters**: 33,944,257

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs         | 50    |
| Batch Size     | 16    |
| Learning Rate  | 1e-4  |
| Optimizer      | Adam  |
| Loss Function  | MSELoss |
| LR Scheduler   | ReduceLROnPlateau (patience=5, factor=0.5) |

## Training Results

- **Best Validation Loss**: 0.0323 (Epoch 2)

### Test Set Performance

| Metric | Value |
|--------|-------|
| MSE    | 0.0416 |
| RMSE   | 0.2040 |
| MAE    | 0.1685 |
| **R²** | **-0.4843** |

### Prediction Examples

| Actual | Predicted |
|--------|-----------|
| 1.280  | 0.873     |
| 1.113  | 0.857     |
| 0.931  | 0.861     |
| 0.847  | 0.875     |
| 0.929  | 0.864     |
| 0.726  | 0.893     |
| 1.010  | 0.846     |
| 0.681  | 0.874     |
| 1.105  | 0.864     |
| 1.219  | 0.863     |

## ⚠️ Issues Requiring Improvement

### 1. Poor Model Prediction Performance
- **R² = -0.4843**: Negative value indicates the model performs worse than simply predicting the mean
- Predictions are clustered in a narrow range (0.85-0.89), while actual values span 0.68-1.28
- The model has essentially failed to learn the relationship between input and output

### 2. Premature Convergence / Overfitting Signs
- Best model achieved at Epoch 2, with no significant improvement in validation loss afterward
- Abnormally high training loss in Epoch 1 (49.6499), suggesting potential data preprocessing or model initialization issues

### 3. Insufficient Data
- Only 240 samples for a deep learning model with 34 million parameters is severely inadequate
- Training set of only 192 samples leads to high overfitting risk

### 4. Potential Improvements
- [ ] **Data Augmentation**: Rotation, flipping, scaling to expand training data
- [ ] **Transfer Learning**: Use pretrained models to reduce dependency on large datasets
- [ ] **Model Simplification**: Reduce model parameters to match data scale
- [ ] **Cross-Validation**: Use K-Fold cross-validation to better utilize limited data
- [ ] **Feature Engineering**: Consider extracting image features instead of end-to-end learning
- [ ] **Regularization**: Add Dropout, L2 regularization to prevent overfitting
- [ ] **Learning Rate Tuning**: Experiment with different learning rate strategies
- [ ] **Loss Function**: Try Huber Loss or other loss functions more robust to outliers

## Usage

```bash
python train.py
```