# src/config.py

# Change ONLY this when you want different number of slices/channels
N_SLICES = 3  # Reduced back to 3 to reduce overfitting (was 5, too many parameters for small dataset)

IMG_SIZE = 256
BATCH_SIZE = 16  # Increased from 8 (use 32 if GPU memory allows)
RANDOM_STATE = 42
NUM_WORKERS = 0

EPOCHS = 50  # CRITICAL: Increased from 2 (minimum 80, can go to 150-200)
LR = 1e-4  # Learning rate for full fine-tuning (all layers trainable)
WEIGHT_DECAY = 5e-4  # Increased from 1e-4 to reduce overfitting (5x stronger regularization)

HUBER_BETA = 0.5

DATA_DIR = "data/Sagittal_T1_FLAIR"
XLSX_PATH = "data/metadata.xlsx"

# Model saving configuration
# Set MODEL_NAME to None to auto-generate timestamp-based name
# Or set a custom name like "experiment_1", "baseline", "augmented_v1", etc.
MODEL_NAME = 'regularized_v1'  # None = auto-generate, or set custom name like "run_1", "experiment_2", etc.

# All outputs (model, results, visualizations) will be saved in one folder
# If MODEL_NAME is None: auto-generate timestamp-based folder name
# If MODEL_NAME is set: use that name as folder name
import datetime
import os

if MODEL_NAME is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"output_{timestamp}"
    model_file_name = f"best_model_{timestamp}.pth"
else:
    OUTPUT_DIR = MODEL_NAME
    model_file_name = f"best_model_{MODEL_NAME}.pth"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model file path (inside output folder)
BEST_PATH = os.path.join(OUTPUT_DIR, model_file_name)

# Results directory (same as output folder)
RESULTS_DIR = OUTPUT_DIR

# -------------------------
# HIGH IMPACT IMPROVEMENTS
# -------------------------

# ROI center crop (applied BEFORE resize). 384 works well for 512x512 sagittal.
# Try 384 first. If spine is too small, use 448. If too much background, use 320.
CENTER_CROP = 384

# Slice jitter (training only): randomly shift the mid slice by +/- this amount.
# Increased for more augmentation (was 1, now 2)
SLICE_JITTER = 2  # Increased from 1 (try 2 or 3)

# Robust intensity normalization per slice
CLIP_LO = 0.5
CLIP_HI = 99.5

# Optional: use robust normalization instead of plain z-score
# CRITICAL: Enable robust normalization (better for MRI images)
# robust: (x - median) / (IQR + eps)  [often more stable for MRI]
USE_ROBUST_NORM = True  # Changed from False (recommended for medical images)

# Test script mode: Set to True to only evaluate saved model (skip training)
# When True: Loads model from BEST_PATH and shows results only (no plots)
TESTSCRIPT = True  # Set to True to run test-only mode

# Override BEST_PATH for test script if you have a model in root directory
# Set to None to use the default path based on MODEL_NAME
# Example: TEST_MODEL_PATH = "best_model.pth"  # Use model from root directory
TEST_MODEL_PATH = None  # None = use default BEST_PATH, or set custom path like "best_model.pth" or "Modelresults1/best_model_Modelresults1.pth"

# Override BEST_PATH if TEST_MODEL_PATH is set and TESTSCRIPT is True
if TESTSCRIPT and TEST_MODEL_PATH is not None:
    BEST_PATH = TEST_MODEL_PATH
