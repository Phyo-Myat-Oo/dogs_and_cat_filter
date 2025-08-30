#!/usr/bin/env python3
"""
Simple training script combining functionality from all notebooks
Minimal dependencies - run with: python simple_train_minimal.py
"""

import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['tensorflow', 'numpy', 'opencv-python', 'matplotlib', 'tqdm']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'tensorflow':
                import tensorflow as tf
            elif package == 'numpy':
                import numpy as np
            elif package == 'matplotlib':
                import matplotlib.pyplot as plt
            elif package == 'tqdm':
                import tqdm
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def main():
    """Main function"""
    print("=== Simple U-Net Training Script ===")
    
    # Check requirements
    if not check_requirements():
        print("Please install missing packages first.")
        sys.exit(1)
    
    # Check dataset
    data_dir = 'dataset/cat_and_dog_dataset'
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}")
        print("Available datasets:")
        if os.path.exists('dataset'):
            for item in os.listdir('dataset'):
                if os.path.isdir(os.path.join('dataset', item)):
                    print(f"  - dataset/{item}")
        sys.exit(1)
    
    # Check required subdirectories
    image_dir = os.path.join(data_dir, 'JPEGImages')
    mask_dir = os.path.join(data_dir, 'SegmentationClass')
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Required directories not found:")
        print(f"  Images: {image_dir} ({'exists' if os.path.exists(image_dir) else 'missing'})")
        print(f"  Masks: {mask_dir} ({'exists' if os.path.exists(mask_dir) else 'missing'})")
        sys.exit(1)
    
    # Import and run training
    try:
        from simple_train import train_model
        train_model()
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()