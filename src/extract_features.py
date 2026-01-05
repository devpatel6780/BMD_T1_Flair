"""
Extract CNN features from MRI images using pretrained ResNet18.
Features will be used for traditional ML models (XGBoost, RF, etc.).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    DATA_DIR, XLSX_PATH, RANDOM_STATE,
    IMG_SIZE, N_SLICES, CENTER_CROP, CLIP_LO, CLIP_HI, USE_ROBUST_NORM,
    MODEL_NAME, OUTPUT_DIR, BEST_PATH
)
from src.DataLoader import (
    load_metadata, get_patient_files, MultiScanDataset, _Preprocessor
)
from src.Model_Transfer import ResNetTransfer


class FeatureExtractor(nn.Module):
    """Extract features from trained CNN model (task-specific features)"""
    
    def __init__(self, trained_model_path, in_ch=3, dropout=0.3):
        super().__init__()
        
        # Load the trained ResNetTransfer model
        print(f"Loading trained model from: {trained_model_path}")
        model = ResNetTransfer(in_ch=in_ch, dropout=dropout)
        
        # Load trained weights
        checkpoint = torch.load(trained_model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Extract only the feature backbone (before regression head)
        self.features = model.features
        
        # Freeze all layers (no training, just feature extraction)
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()  # Set to eval mode
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: [B, 512]
        return x


def extract_features_from_dataset(model, dataset, device, batch_size=16):
    """Extract features from a dataset"""
    model.eval()
    
    features_list = []
    pids_list = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    with torch.no_grad():
        for images, labels, pids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)  # [B, 512]
            
            features_list.append(features.cpu().numpy())
            pids_list.append(pids.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    pids = np.concatenate(pids_list, axis=0)
    
    return features, pids


def aggregate_features_per_patient(features, pids):
    """Average features across scans for each patient"""
    pids = pids.astype(np.int64)
    unique_pids = np.unique(pids)
    
    patient_features = {}
    for pid in unique_pids:
        mask = (pids == pid)
        patient_features[pid] = np.mean(features[mask], axis=0)
    
    # Return in same order as unique_pids
    feature_array = np.array([patient_features[pid] for pid in unique_pids])
    return feature_array, unique_pids


def main():
    print("=" * 70)
    print("CNN Feature Extraction for Traditional ML Models")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    bmd_dict = load_metadata(XLSX_PATH)
    patient_files = get_patient_files(DATA_DIR)
    
    valid_patients = [pid for pid in patient_files.keys() if pid in bmd_dict]
    
    print(f"Total patients found: {len(patient_files)}")
    print(f"Valid patients with labels: {len(valid_patients)}")
    
    # Same patient-level split as CNN training
    train_pids, temp_pids = train_test_split(
        valid_patients, test_size=0.2, random_state=RANDOM_STATE
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=RANDOM_STATE
    )
    
    print("\n===== SPLIT SUMMARY (patient-level) =====")
    print(f"Train patients: {len(train_pids)}")
    print(f"Val patients  : {len(val_pids)}")
    print(f"Test patients : {len(test_pids)}")
    
    # Preprocessor (same as CNN, but no jitter for eval)
    preprocessor = _Preprocessor(
        img_size=IMG_SIZE,
        n_slices=N_SLICES,
        center_crop=CENTER_CROP,
        clip_lo=CLIP_LO,
        clip_hi=CLIP_HI,
        use_robust_norm=USE_ROBUST_NORM,
        slice_jitter=0,  # No jitter for feature extraction
    )
    
    # Create datasets (use all scans per patient)
    print("\nCreating datasets...")
    train_dataset = MultiScanDataset(
        train_pids, patient_files, bmd_dict,
        preproc=preprocessor, y_mean=0.0, y_std=1.0  # Not used for features
    )
    val_dataset = MultiScanDataset(
        val_pids, patient_files, bmd_dict,
        preproc=preprocessor, y_mean=0.0, y_std=1.0
    )
    test_dataset = MultiScanDataset(
        test_pids, patient_files, bmd_dict,
        preproc=preprocessor, y_mean=0.0, y_std=1.0
    )
    
    # Create feature extractor from TRAINED model
    print("\n" + "=" * 70)
    print("Loading TRAINED CNN model for feature extraction...")
    print("=" * 70)
    print(f"Model path: {BEST_PATH}")
    
    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {BEST_PATH}\n"
            f"Please train the CNN model first using: python train.py"
        )
    
    feature_extractor = FeatureExtractor(
        trained_model_path=BEST_PATH,
        in_ch=N_SLICES,
        dropout=0.3
    ).to(device)
    
    # Get feature dimension
    dummy_input = torch.randn(1, N_SLICES, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        dummy_features = feature_extractor(dummy_input)
    feature_dim = dummy_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Extract features
    print("\n" + "=" * 70)
    print("Extracting features from datasets...")
    print("=" * 70)
    
    train_features, train_pids = extract_features_from_dataset(
        feature_extractor, train_dataset, device, batch_size=16
    )
    print(f"\nTrain features shape: {train_features.shape}")
    
    val_features, val_pids = extract_features_from_dataset(
        feature_extractor, val_dataset, device, batch_size=16
    )
    print(f"Val features shape: {val_features.shape}")
    
    test_features, test_pids = extract_features_from_dataset(
        feature_extractor, test_dataset, device, batch_size=16
    )
    print(f"Test features shape: {test_features.shape}")
    
    # Aggregate per patient (average across scans)
    print("\nAggregating features per patient...")
    train_feat_p, train_pids_p = aggregate_features_per_patient(train_features, train_pids)
    val_feat_p, val_pids_p = aggregate_features_per_patient(val_features, val_pids)
    test_feat_p, test_pids_p = aggregate_features_per_patient(test_features, test_pids)
    
    print(f"Train features (per-patient): {train_feat_p.shape}")
    print(f"Val features (per-patient): {val_feat_p.shape}")
    print(f"Test features (per-patient): {test_feat_p.shape}")
    
    # Create output directory
    output_dir = Path("features")
    output_dir.mkdir(exist_ok=True)
    
    # Save features
    print("\n" + "=" * 70)
    print("Saving features...")
    print("=" * 70)
    
    np.save(output_dir / "features_train.npy", train_feat_p)
    np.save(output_dir / "patient_ids_train.npy", train_pids_p)
    
    np.save(output_dir / "features_val.npy", val_feat_p)
    np.save(output_dir / "patient_ids_val.npy", val_pids_p)
    
    np.save(output_dir / "features_test.npy", test_feat_p)
    np.save(output_dir / "patient_ids_test.npy", test_pids_p)
    
    # Save labels (per-patient)
    train_labels = np.array([float(bmd_dict[pid]) for pid in train_pids_p])
    val_labels = np.array([float(bmd_dict[pid]) for pid in val_pids_p])
    test_labels = np.array([float(bmd_dict[pid]) for pid in test_pids_p])
    
    np.save(output_dir / "labels_train.npy", train_labels)
    np.save(output_dir / "labels_val.npy", val_labels)
    np.save(output_dir / "labels_test.npy", test_labels)
    
    # Save split info
    split_info = {
        'train_pids': train_pids,
        'val_pids': val_pids,
        'test_pids': test_pids,
        'random_state': RANDOM_STATE,
        'feature_dim': feature_dim,
        'n_slices': N_SLICES,
        'img_size': IMG_SIZE,
    }
    np.save(output_dir / "split_info.npy", split_info, allow_pickle=True)
    
    print(f"\nAll features saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  - features_train.npy: {train_feat_p.shape}")
    print(f"  - features_val.npy: {val_feat_p.shape}")
    print(f"  - features_test.npy: {test_feat_p.shape}")
    print(f"  - labels_*.npy: Corresponding BMD labels")
    print(f"  - patient_ids_*.npy: Patient IDs")
    print(f"  - split_info.npy: Split information")
    
    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)
    print("\nNext step: Run train_ml_models.py to train ML models on features")


if __name__ == "__main__":
    main()

