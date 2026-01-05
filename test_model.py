#!/usr/bin/env python3
"""
Helper script to test different saved models easily.
Usage: python test_model.py <model_path>
Example: python test_model.py Modelresults1/best_model_Modelresults1.pth
"""

import os
import sys
import glob

def list_available_models():
    """List all available model files"""
    print("=" * 70)
    print("Available Models:")
    print("=" * 70)
    
    patterns = [
        "best_model*.pth",
        "*/best_model*.pth",
        "**/best_model*.pth"
    ]
    
    models = []
    for pattern in patterns:
        models.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    models = sorted(set(models))
    
    if not models:
        print("No model files found!")
        return []
    
    for i, model_path in enumerate(models, 1):
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"{i:2d}. {model_path} ({size:.1f} MB)")
    
    print("=" * 70)
    return models


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("\nAvailable models:")
            list_available_models()
            sys.exit(1)
        
        print(f"Testing model: {model_path}")
        
        # Update config to use this model
        config_path = "src/config.py"
        
        # Read config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Update TEST_MODEL_PATH
        import re
        # Find and replace TEST_MODEL_PATH line
        pattern = r"TEST_MODEL_PATH\s*=\s*[^\n]+"
        replacement = f"TEST_MODEL_PATH = '{model_path}'"
        config_content = re.sub(pattern, replacement, config_content)
        
        # Ensure TESTSCRIPT is True
        config_content = re.sub(
            r"TESTSCRIPT\s*=\s*False",
            "TESTSCRIPT = True",
            config_content
        )
        
        # Write back
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Updated config.py to use: {model_path}")
        print("Now run: python train.py")
        
    else:
        print("Usage: python test_model.py <model_path>")
        print("\nExample:")
        print("  python test_model.py best_model.pth")
        print("  python test_model.py Modelresults1/best_model_Modelresults1.pth")
        print("  python test_model.py full_finetune/best_model_full_finetune.pth")
        print("\nOr see all available models:")
        print("  python test_model.py --list")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--list":
            list_available_models()


if __name__ == "__main__":
    main()

