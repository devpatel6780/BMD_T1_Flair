import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2


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
        depth = data.shape[2]
        middle_slice = data[:, :, depth // 2]

        # Resize to target size
        middle_slice = cv2.resize(middle_slice, (self.img_size, self.img_size))

        # Normalize (Min-Max scaling to [0, 1])
        min_val = middle_slice.min()
        max_val = middle_slice.max()
        if max_val - min_val > 0:
            middle_slice = (middle_slice - min_val) / (max_val - min_val)
        else:
            middle_slice = np.zeros_like(middle_slice)

        # Convert to PyTorch tensor [1, H, W]
        image_tensor = torch.FloatTensor(middle_slice).unsqueeze(0)
        label_tensor = torch.FloatTensor([self.labels[idx]])

        return image_tensor, label_tensor.squeeze()


def load_metadata(xlsx_path):
    """Load metadata.xlsx, return ID to BMD mapping"""
    df = pd.read_excel(xlsx_path)
    return dict(zip(df['ID'].astype(int), df['BMD'].astype(float)))


def get_patient_files(data_dir):
    """
    Scan directory, build patient ID to file path mapping
    Filename format: {PatientID}_{Date}_{Description}.nii.gz
    """
    patient_files = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.nii.gz'):
            # Parse patient ID (number before first _ or @)
            try:
                # Handle filenames like "368@_..." or "368_..."
                first_part = filename.split('_')[0]
                # Remove @ if present
                first_part = first_part.replace('@', '')
                patient_id = int(first_part)
                file_path = os.path.join(data_dir, filename)

                if patient_id not in patient_files:
                    patient_files[patient_id] = []
                patient_files[patient_id].append(file_path)
            except ValueError:
                print(f"Warning: Cannot parse filename {filename}")
                continue

    return patient_files


def create_dataloaders(data_dir, xlsx_path, batch_size=16, img_size=256, random_state=42):
    """
    Create train/val/test DataLoaders

    Args:
        data_dir: NIfTI files directory
        xlsx_path: metadata.xlsx path
        batch_size: Batch size
        img_size: Image size
        random_state: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load metadata
    bmd_dict = load_metadata(xlsx_path)

    # Get file mapping
    patient_files = get_patient_files(data_dir)

    # Build (file_path, bmd) pairs
    file_paths = []
    labels = []

    for patient_id, files in patient_files.items():
        if patient_id in bmd_dict:
            bmd = bmd_dict[patient_id]
            for file_path in files:
                file_paths.append(file_path)
                labels.append(bmd)
        else:
            print(f"Warning: Patient ID {patient_id} not found in metadata")

    print(f"Total samples loaded: {len(file_paths)}")

    # 8:1:1 split
    # First 80:20 split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=random_state
    )

    # Then split 20% into 10:10 (i.e., 50:50)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=random_state
    )

    print(f"Train set: {len(train_files)} samples")
    print(f"Val set: {len(val_files)} samples")
    print(f"Test set: {len(test_files)} samples")

    # Create Datasets
    train_dataset = BMDDataset(train_files, train_labels, img_size)
    val_dataset = BMDDataset(val_files, val_labels, img_size)
    test_dataset = BMDDataset(test_files, test_labels, img_size)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

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
