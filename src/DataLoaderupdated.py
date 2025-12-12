import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import re


class BMDDataset(Dataset):
    """BMD prediction dataset"""

    def __init__(self, file_paths, labels, img_size=256):
        """
        Args:
            file_paths: List of NIfTI file paths
            labels: List of BMD labels
            img_size: Output image size
        """
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load NIfTI file
        nii_img = nib.load(self.file_paths[idx])
        data = nii_img.get_fdata()

        # Get middle slice
        # --- 2.5D: stack 3 neighboring slices as channels ---
        depth = data.shape[2]
        mid = depth // 2

        # clamp indices so we don't go out of range
        idxs = [max(0, mid - 1), mid, min(depth - 1, mid + 1)]

        slices = []
        for k in idxs:
            sl = data[:, :, k]
            sl = cv2.resize(sl, (self.img_size, self.img_size))

            # Z-score per slice
            mean_val = float(sl.mean())
            std_val  = float(sl.std())
            sl = (sl - mean_val) / (std_val + 1e-6)

            slices.append(sl)

        # shape: (3, H, W)
        image_tensor = torch.from_numpy(np.stack(slices, axis=0)).float()
        label_tensor = torch.FloatTensor([self.labels[idx]])

        return image_tensor, label_tensor.squeeze()


def load_metadata(xlsx_path):
    """Load metadata.xlsx, return ID to BMD mapping"""
    df = pd.read_excel(xlsx_path)
    return dict(zip(df['ID'].astype(int), df['BMD'].astype(float)))


def get_patient_files(data_dir):
    """
    Scan directory, build patient ID -> list of file paths mapping.
    Robust to filenames like:
      368@_20200805_....nii.gz
      368_20200805_....nii.gz
    """
    patient_files = {}

    for filename in os.listdir(data_dir):
        if not filename.endswith(".nii.gz"):
            continue

        # Patient id is the first token before '_' but may contain '@' etc.
        first_token = filename.split("_")[0]
        m = re.search(r"\d+", first_token)   # extract digits anywhere in token
        if not m:
            print(f"Warning: Cannot parse patient ID from {filename}")
            continue

        patient_id = int(m.group())
        file_path = os.path.join(data_dir, filename)

        patient_files.setdefault(patient_id, []).append(file_path)

    return patient_files



def create_dataloaders(data_dir, xlsx_path, batch_size=16, img_size=256, random_state=42):
    """
    Create train/val/test DataLoaders using PATIENT-LEVEL split (no leakage).
    """
    # Load metadata: patient_id -> bmd
    bmd_dict = load_metadata(xlsx_path)

    # patient_id -> [file paths]
    patient_files = get_patient_files(data_dir)

    # Keep only patients that exist in metadata and have >=1 file
    valid_patients = [pid for pid in patient_files.keys() if pid in bmd_dict]

    missing_in_metadata = [pid for pid in patient_files.keys() if pid not in bmd_dict]
    if missing_in_metadata:
        print(f"Warning: {len(missing_in_metadata)} patient IDs not found in metadata (ignored).")

    if len(valid_patients) < 3:
        raise ValueError(f"Not enough valid patients to split. Found {len(valid_patients)}")

    print(f"Total patients found in folder: {len(patient_files)}")
    print(f"Valid patients with labels     : {len(valid_patients)}")

    # ---- PATIENT-LEVEL 80/10/10 split ----
    train_pids, temp_pids = train_test_split(
        valid_patients, test_size=0.2, random_state=random_state
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=random_state
    )

    # Expand patient IDs -> (file_path, label)
    def expand(pids):
        paths, ys = [], []
        for pid in pids:
            y = bmd_dict[pid]
            for fp in patient_files[pid]:
                paths.append(fp)
                ys.append(y)
        return paths, ys

    train_files, train_labels = expand(train_pids)
    val_files, val_labels     = expand(val_pids)
    test_files, test_labels   = expand(test_pids)

    print("\n===== SPLIT SUMMARY (patient-level) =====")
    print(f"Train patients: {len(train_pids)} | samples: {len(train_files)}")
    print(f"Val patients  : {len(val_pids)} | samples: {len(val_files)}")
    print(f"Test patients : {len(test_pids)} | samples: {len(test_files)}")

    # (Optional but useful) sanity check: no patient overlap
    assert set(train_pids).isdisjoint(val_pids)
    assert set(train_pids).isdisjoint(test_pids)
    assert set(val_pids).isdisjoint(test_pids)

    # Create datasets
    train_dataset = BMDDataset(train_files, train_labels, img_size)
    val_dataset   = BMDDataset(val_files, val_labels, img_size)
    test_dataset  = BMDDataset(test_files, test_labels, img_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    # Test data loading
    data_dir = 'data/Sagittal_T1_FLAIR'
    xlsx_path = 'data/metadata.xlsx'

    train_loader, val_loader, test_loader = create_dataloaders(data_dir, xlsx_path)

    # Test one batch
    for images, labels in train_loader:
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Label range: {labels.min():.3f} - {labels.max():.3f}")
        break
