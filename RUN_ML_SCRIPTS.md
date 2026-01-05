# How to Run ML Scripts

## Important: Run from Project Root

**Always run scripts from the project root directory**, not from inside `src/`:

```bash
# ✅ CORRECT: From project root
cd /home/dev/Downloads/BMD_T1_Flair-main
python src/extract_features.py

# ❌ WRONG: From inside src/
cd src/
python extract_features.py  # This will fail!
```

---

## Quick Commands

### Step 1: Extract Features
```bash
# Make sure you're in project root
cd /home/dev/Downloads/BMD_T1_Flair-main

# Run feature extraction
python src/extract_features.py
```

### Step 2: Train ML Models
```bash
# Still in project root
python src/train_ml_models.py
```

### Step 3: Compare (Optional)
```bash
# Still in project root
python src/compare_ml_cnn.py
```

---

## Why This Matters

The scripts import from `src.config`, `src.DataLoader`, etc.
- ✅ Works when run from project root: `python src/extract_features.py`
- ❌ Fails when run from src/: `cd src && python extract_features.py`

---

## Quick Fix

If you're in `src/` directory, just go back:
```bash
cd ..  # Go to project root
python src/extract_features.py
```

