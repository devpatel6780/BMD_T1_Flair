# ########################################
# # best r2 0.32
# ########################################


# import os
# import pandas as pd
# import nibabel as nib
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# import cv2
# import re


# class BMDDataset(Dataset):
#     """
#     BMD prediction dataset
#     - Reads .nii.gz
#     - Builds 2.5D 3-slice input: [mid-1, mid, mid+1] as channels -> [3,H,W]
#     - Per-slice z-score
#     - Target normalization (using train mean/std)
#     """

#     def __init__(self, file_paths, labels, img_size=256, y_mean=None, y_std=None):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.img_size = int(img_size)

#         # If provided, normalize y: (y - mean)/std
#         self.y_mean = float(y_mean) if y_mean is not None else None
#         self.y_std = float(y_std) if y_std is not None else None

#     def __len__(self):
#         return len(self.file_paths)

#     @staticmethod
#     def _safe_zscore(x: np.ndarray) -> np.ndarray:
#         m = float(x.mean())
#         s = float(x.std())
#         return (x - m) / (s + 1e-6)

#     @staticmethod
#     def _clip_percentile(x: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
#         a = np.percentile(x, lo)
#         b = np.percentile(x, hi)
#         return np.clip(x, a, b)

#     def __getitem__(self, idx):
#         # Load NIfTI file
#         nii_img = nib.load(self.file_paths[idx])
#         data = nii_img.get_fdata().astype(np.float32)

#         # ---- Choose middle slice index along depth axis (assume data[:,:,k]) ----
#         depth = data.shape[2]
#         mid = depth // 2
#         idxs = [max(0, mid - 1), mid, min(depth - 1, mid + 1)]
        
#         slices = []
#         for k in idxs:
#             sl = data[:, :, k]

#             # Robust clip to reduce outliers (helps medical images)
#             sl = self._clip_percentile(sl, 0.5, 99.5)

#             # Resize to model size
#             sl = cv2.resize(sl, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

#             # Per-slice z-score
#             sl = self._safe_zscore(sl)
#             slices.append(sl)

#         # [3,H,W]
#         image_tensor = torch.from_numpy(np.stack(slices, axis=0)).float()

#         y = float(self.labels[idx])
#         if (self.y_mean is not None) and (self.y_std is not None):
#             y = (y - self.y_mean) / (self.y_std + 1e-8)

#         label_tensor = torch.tensor(y, dtype=torch.float32)
#         return image_tensor, label_tensor


# def load_metadata(xlsx_path):
#     """Load metadata.xlsx, return ID -> BMD mapping"""
#     df = pd.read_excel(xlsx_path)
#     # robust types
#     ids = df['ID'].astype(int).tolist()
#     bmd = df['BMD'].astype(float).tolist()
#     return dict(zip(ids, bmd))


# def get_patient_files(data_dir):
#     """
#     Scan directory, build patient ID -> list of file paths mapping.
#     Robust to filenames like:
#       368@_20200805_....nii.gz
#       368_20200805_....nii.gz
#     """
#     patient_files = {}

#     for filename in os.listdir(data_dir):
#         if not filename.endswith(".nii.gz"):
#             continue

#         first_token = filename.split("_")[0]
#         m = re.search(r"\d+", first_token)  # extract digits anywhere
#         if not m:
#             print(f"Warning: Cannot parse patient ID from {filename}")
#             continue

#         patient_id = int(m.group())
#         file_path = os.path.join(data_dir, filename)
#         patient_files.setdefault(patient_id, []).append(file_path)

#     return patient_files


# def create_dataloaders(
#     data_dir,
#     xlsx_path,
#     batch_size=16,
#     img_size=256,
#     random_state=42,
#     num_workers=0
# ):
#     """
#     Create train/val/test DataLoaders using PATIENT-LEVEL split (no leakage).
#     Also computes train target mean/std and applies it to all splits.
#     Returns: train_loader, val_loader, test_loader, (y_mean, y_std)
#     """
#     bmd_dict = load_metadata(xlsx_path)
#     patient_files = get_patient_files(data_dir)

#     valid_patients = [pid for pid in patient_files.keys() if pid in bmd_dict]

#     missing_in_metadata = [pid for pid in patient_files.keys() if pid not in bmd_dict]
#     if missing_in_metadata:
#         print(f"Warning: {len(missing_in_metadata)} patient IDs not found in metadata (ignored).")

#     if len(valid_patients) < 3:
#         raise ValueError(f"Not enough valid patients to split. Found {len(valid_patients)}")

#     print(f"Total patients found in folder: {len(patient_files)}")
#     print(f"Valid patients with labels     : {len(valid_patients)}")

#     # ---- Patient-level split ----
#     train_pids, temp_pids = train_test_split(
#         valid_patients, test_size=0.2, random_state=random_state
#     )
#     val_pids, test_pids = train_test_split(
#         temp_pids, test_size=0.5, random_state=random_state
#     )

#     def expand(pids):
#         paths, ys = [], []
#         for pid in pids:
#             y = float(bmd_dict[pid])
#             for fp in patient_files[pid]:
#                 paths.append(fp)
#                 ys.append(y)
#         return paths, ys

#     train_files, train_labels = expand(train_pids)
#     val_files, val_labels     = expand(val_pids)
#     test_files, test_labels   = expand(test_pids)

#     print("\n===== SPLIT SUMMARY (patient-level) =====")
#     print(f"Train patients: {len(train_pids)} | samples: {len(train_files)}")
#     print(f"Val patients  : {len(val_pids)} | samples: {len(val_files)}")
#     print(f"Test patients : {len(test_pids)} | samples: {len(test_files)}")

#     # Sanity: no overlap
#     assert set(train_pids).isdisjoint(val_pids)
#     assert set(train_pids).isdisjoint(test_pids)
#     assert set(val_pids).isdisjoint(test_pids)

#     # ---- Target normalization stats from TRAIN labels only ----
#     y_train = np.array(train_labels, dtype=np.float32)
#     y_mean = float(y_train.mean())
#     y_std = float(y_train.std() + 1e-8)

#     print("\n===== TARGET STATS (train only) =====")
#     print(f"y_mean: {y_mean:.6f}")
#     print(f"y_std : {y_std:.6f}")

#     # Datasets (apply same y_mean/y_std to ALL splits)
#     train_dataset = BMDDataset(train_files, train_labels, img_size=img_size, y_mean=y_mean, y_std=y_std)
#     val_dataset   = BMDDataset(val_files,   val_labels,   img_size=img_size, y_mean=y_mean, y_std=y_std)
#     test_dataset  = BMDDataset(test_files,  test_labels,  img_size=img_size, y_mean=y_mean, y_std=y_std)

#     pin_memory = torch.cuda.is_available()

#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=pin_memory
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=pin_memory
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=pin_memory
#     )

#     return train_loader, val_loader, test_loader, (y_mean, y_std)


# if __name__ == '__main__':
#     data_dir = 'data/Sagittal_T1_FLAIR'
#     xlsx_path = 'data/metadata.xlsx'

#     train_loader, val_loader, test_loader, stats = create_dataloaders(
#         data_dir, xlsx_path, batch_size=8, img_size=256
#     )
#     y_mean, y_std = stats
#     print("Stats:", y_mean, y_std)

#     for images, labels in train_loader:
#         print(f"Image shape: {images.shape}")  # [B,3,256,256]
#         print(f"Label shape: {labels.shape}")  # [B]
#         print(f"Label (normalized) range: {labels.min().item():.3f} - {labels.max().item():.3f}")
#         break



# src/DataLoader.py

import os
import re
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def get_sagittal_axis(nii_img: nib.Nifti1Image) -> int:
    """
    Auto-detect sagittal axis from affine orientation.
    Sagittal plane is orthogonal to the Left-Right axis => code 'L' or 'R'.
    """
    axcodes = nib.aff2axcodes(nii_img.affine)
    for i, c in enumerate(axcodes):
        if c in ("L", "R"):
            return i
    raise RuntimeError(f"No sagittal (L/R) axis found in orientation codes: {axcodes}")


class _Preprocessor:
    """
    Shared preprocessing logic:
    - load .nii.gz
    - auto-detect sagittal axis
    - choose N slices around mid (+ optional jitter for train)
    - clip -> center-crop -> resize -> robust norm / zscore
    """

    def __init__(
        self,
        img_size=256,
        n_slices=3,
        center_crop=384,
        clip_lo=0.5,
        clip_hi=99.5,
        use_robust_norm=True,
        slice_jitter=0,
    ):
        self.img_size = int(img_size)
        self.n_slices = int(n_slices)
        self.center_crop = int(center_crop) if center_crop is not None else None
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.use_robust_norm = bool(use_robust_norm)
        self.slice_jitter = int(slice_jitter)

        if self.n_slices < 1:
            raise ValueError("n_slices must be >= 1")

    @staticmethod
    def _clip_percentile(x: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
        if not np.isfinite(x).any():
            return np.zeros_like(x, dtype=np.float32)
        finite = x[np.isfinite(x)]
        a = np.percentile(finite, lo)
        b = np.percentile(finite, hi)
        if b <= a:
            return np.clip(x, a, a).astype(np.float32)
        return np.clip(x, a, b).astype(np.float32)

    @staticmethod
    def _robust_norm(x: np.ndarray) -> np.ndarray:
        med = float(np.median(x))
        q1 = float(np.percentile(x, 25))
        q3 = float(np.percentile(x, 75))
        iqr = q3 - q1
        return ((x - med) / (iqr + 1e-6)).astype(np.float32)

    @staticmethod
    def _safe_zscore(x: np.ndarray) -> np.ndarray:
        m = float(np.mean(x))
        s = float(np.std(x))
        return ((x - m) / (s + 1e-6)).astype(np.float32)

    @staticmethod
    def _center_crop2d(x: np.ndarray, crop: int) -> np.ndarray:
        if crop is None:
            return x
        h, w = x.shape[:2]
        crop = int(crop)
        crop = min(crop, h, w)

        cy = h // 2
        cx = w // 2
        y1 = max(0, cy - crop // 2)
        x1 = max(0, cx - crop // 2)
        y2 = min(h, y1 + crop)
        x2 = min(w, x1 + crop)

        if (y2 - y1) < crop:
            y1 = max(0, y2 - crop)
        if (x2 - x1) < crop:
            x1 = max(0, x2 - crop)

        return x[y1:y2, x1:x2]

    @staticmethod
    def _choose_slice_indices(depth: int, n_slices: int, mid: int):
        mid = max(0, min(depth - 1, int(mid)))
        half = n_slices // 2
        idxs = [mid + (i - half) for i in range(n_slices)]
        idxs = [min(depth - 1, max(0, k)) for k in idxs]
        return idxs

    @staticmethod
    def _slice2d(data: np.ndarray, axis: int, k: int) -> np.ndarray:
        if axis == 0:
            return data[k, :, :]
        elif axis == 1:
            return data[:, k, :]
        elif axis == 2:
            return data[:, :, k]
        else:
            raise ValueError(f"Unsupported axis: {axis}")

    def __call__(self, file_path: str, is_train: bool) -> torch.Tensor:
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata().astype(np.float32)

        sag_axis = get_sagittal_axis(nii_img)
        depth = int(data.shape[sag_axis])

        mid = depth // 2
        if is_train and self.slice_jitter > 0 and depth > 1:
            j = np.random.randint(-self.slice_jitter, self.slice_jitter + 1)
            mid = max(0, min(depth - 1, mid + j))

        idxs = self._choose_slice_indices(depth, self.n_slices, mid)

        slices = []
        for k in idxs:
            sl = self._slice2d(data, sag_axis, int(k))

            sl = self._clip_percentile(sl, self.clip_lo, self.clip_hi)
            sl = self._center_crop2d(sl, self.center_crop)
            sl = cv2.resize(sl, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            if self.use_robust_norm:
                sl = self._robust_norm(sl)
            else:
                sl = self._safe_zscore(sl)

            if not np.isfinite(sl).all():
                sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            slices.append(sl)

        x = torch.from_numpy(np.stack(slices, axis=0)).float()  # [C,H,W]
        return x


class PatientOneSampleDataset(Dataset):
    """
    1 item == 1 patient.
    - Train: picks a random scan each time (fixes label duplication bias)
    - Val: picks a deterministic scan (middle of sorted list)
    Returns: (image[C,H,W], y_norm, pid)
    """

    def __init__(
        self,
        patient_ids,
        patient_files,   # dict pid -> list[filepaths]
        bmd_dict,        # dict pid -> label
        preproc: _Preprocessor,
        y_mean: float,
        y_std: float,
        is_train: bool,
    ):
        self.pids = list(patient_ids)
        self.patient_files = patient_files
        self.bmd_dict = bmd_dict
        self.preproc = preproc
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.is_train = bool(is_train)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = int(self.pids[idx])
        files = self.patient_files[pid]
        if self.is_train:
            fp = files[np.random.randint(0, len(files))]
        else:
            files_sorted = sorted(files)
            fp = files_sorted[len(files_sorted) // 2]

        x = self.preproc(fp, is_train=self.is_train)

        y = float(self.bmd_dict[pid])
        y_norm = (y - self.y_mean) / (self.y_std + 1e-8)

        return x, torch.tensor(y_norm, dtype=torch.float32), torch.tensor(pid, dtype=torch.int64)


class MultiScanDataset(Dataset):
    """
    1 item == 1 scan (file), but also returns pid.
    Used for true per-patient evaluation by averaging preds across scans.
    Returns: (image[C,H,W], y_norm, pid)
    """

    def __init__(
        self,
        patient_ids,
        patient_files,
        bmd_dict,
        preproc: _Preprocessor,
        y_mean: float,
        y_std: float,
    ):
        self.items = []
        for pid in patient_ids:
            pid = int(pid)
            for fp in patient_files[pid]:
                self.items.append((pid, fp))

        self.bmd_dict = bmd_dict
        self.preproc = preproc
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, fp = self.items[idx]
        x = self.preproc(fp, is_train=False)

        y = float(self.bmd_dict[pid])
        y_norm = (y - self.y_mean) / (self.y_std + 1e-8)

        return x, torch.tensor(y_norm, dtype=torch.float32), torch.tensor(pid, dtype=torch.int64)


def load_metadata(xlsx_path):
    df = pd.read_excel(xlsx_path)
    ids = df["ID"].astype(int).tolist()
    bmd = df["BMD"].astype(float).tolist()
    return dict(zip(ids, bmd))


def get_patient_files(data_dir):
    patient_files = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".nii.gz"):
            continue

        first_token = filename.split("_")[0]
        m = re.search(r"\d+", first_token)
        if not m:
            print(f"Warning: Cannot parse patient ID from {filename}")
            continue

        patient_id = int(m.group())
        file_path = os.path.join(data_dir, filename)
        patient_files.setdefault(patient_id, []).append(file_path)

    return patient_files


def create_dataloaders(
    data_dir,
    xlsx_path,
    batch_size=16,
    img_size=256,
    random_state=42,
    num_workers=0,
    n_slices=3,
    center_crop=384,
    clip_lo=0.5,
    clip_hi=99.5,
    use_robust_norm=True,
    slice_jitter=0,
):
    bmd_dict = load_metadata(xlsx_path)
    patient_files = get_patient_files(data_dir)

    valid_patients = [pid for pid in patient_files.keys() if pid in bmd_dict]
    missing_in_metadata = [pid for pid in patient_files.keys() if pid not in bmd_dict]
    if missing_in_metadata:
        print(f"Warning: {len(missing_in_metadata)} patient IDs not found in metadata (ignored).")

    if len(valid_patients) < 3:
        raise ValueError(f"Not enough valid patients to split. Found {len(valid_patients)}")

    print(f"Total patients found in folder: {len(patient_files)}")
    print(f"Valid patients with labels     : {len(valid_patients)}")

    train_pids, temp_pids = train_test_split(
        valid_patients, test_size=0.2, random_state=random_state
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=random_state
    )

    print("\n===== SPLIT SUMMARY (patient-level) =====")
    print(f"Train patients: {len(train_pids)}")
    print(f"Val patients  : {len(val_pids)}")
    print(f"Test patients : {len(test_pids)}")

    assert set(train_pids).isdisjoint(val_pids)
    assert set(train_pids).isdisjoint(test_pids)
    assert set(val_pids).isdisjoint(test_pids)

    # Train target stats computed PER PATIENT (no duplication)
    y_train = np.array([float(bmd_dict[pid]) for pid in train_pids], dtype=np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)

    print("\n===== TARGET STATS (train only, per-patient) =====")
    print(f"y_mean: {y_mean:.6f}")
    print(f"y_std : {y_std:.6f}")
    print(f"n_slices (channels): {int(n_slices)}")
    print(f"center_crop: {int(center_crop)}")
    print(f"slice_jitter(train): {int(slice_jitter)}")
    print(f"robust_norm: {bool(use_robust_norm)}")

    preproc_train = _Preprocessor(
        img_size=img_size,
        n_slices=n_slices,
        center_crop=center_crop,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        use_robust_norm=use_robust_norm,
        slice_jitter=slice_jitter,
    )
    preproc_eval = _Preprocessor(
        img_size=img_size,
        n_slices=n_slices,
        center_crop=center_crop,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        use_robust_norm=use_robust_norm,
        slice_jitter=0,
    )

    # TRAIN: one sample per patient per epoch (random scan selection)
    train_dataset = PatientOneSampleDataset(
        train_pids, patient_files, bmd_dict,
        preproc=preproc_train, y_mean=y_mean, y_std=y_std, is_train=True
    )

    # VAL/TEST: multi-scan dataset so we can average predictions per patient
    val_dataset = MultiScanDataset(
        val_pids, patient_files, bmd_dict,
        preproc=preproc_eval, y_mean=y_mean, y_std=y_std
    )
    test_dataset = MultiScanDataset(
        test_pids, patient_files, bmd_dict,
        preproc=preproc_eval, y_mean=y_mean, y_std=y_std
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # return pid splits too (optional debugging)
    return train_loader, val_loader, test_loader, (y_mean, y_std)
