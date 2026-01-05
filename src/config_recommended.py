# src/config_recommended.py
# RECOMMENDED CONFIGURATION FOR BETTER RESULTS
# Copy these settings to config.py or use as reference

# ===========================================
# PHASE 1: QUICK WINS (Start Here)
# ===========================================

# Change ONLY this when you want different number of slices/channels
N_SLICES = 5  # Increased from 3 (try 5 or 7 for more context)

IMG_SIZE = 256
BATCH_SIZE = 16  # Increased from 8 (use 32 if GPU memory allows)
RANDOM_STATE = 42
NUM_WORKERS = 0

EPOCHS = 100  # CRITICAL: Increased from 2 (minimum 80, can go to 150-200)
LR = 2e-4
WEIGHT_DECAY = 1e-4

HUBER_BETA = 0.5

DATA_DIR = "data/Sagittal_T1_FLAIR"
XLSX_PATH = "data/metadata.xlsx"

# Model saving configuration
MODEL_NAME = 'improved_v1'  # Change for each experiment
# ... (same as config.py)

# ===========================================
# HIGH IMPACT IMPROVEMENTS
# ===========================================

# ROI center crop (applied BEFORE resize)
CENTER_CROP = 384  # Try 448 if spine is too small, 320 if too much background

# Slice jitter (training only): increased for more augmentation
SLICE_JITTER = 2  # Increased from 1 (try 2 or 3)

# Robust intensity normalization per slice
CLIP_LO = 0.5
CLIP_HI = 99.5

# CRITICAL: Enable robust normalization (better for MRI)
USE_ROBUST_NORM = True  # Changed from False (recommended for medical images)

# Test script mode
TESTSCRIPT = False  # Set to True for test-only mode

# ===========================================
# RECOMMENDED SETTINGS SUMMARY
# ===========================================
# EPOCHS: 2 → 100 (MOST IMPORTANT!)
# USE_ROBUST_NORM: False → True (IMPORTANT!)
# BATCH_SIZE: 8 → 16 (if memory allows)
# N_SLICES: 3 → 5 (try 5 or 7)
# SLICE_JITTER: 1 → 2 (more augmentation)
# 
# Expected improvement: R² 0.10 → 0.25-0.35

