#!/usr/bin/env python3

import os
import sys
import glob
import json
import numpy as np
from PIL import Image
import cv2

def test_dataset_structure():
    """Simple test without TensorFlow to check dataset structure"""
    print("=" * 50)
    print("Simple Dataset Structure Test")
    print("=" * 50)
    
    dataset_path = "./dataset/cat_and_dog_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset path found: {dataset_path}")
    
    # Check for different possible structures
    cat_path = os.path.join(dataset_path, "cats/cat")
    dog_path = os.path.join(dataset_path, "dogs/dog") 
    jpeg_path = os.path.join(dataset_path, "JPEGImages")
    seg_path = os.path.join(dataset_path, "SegmentationClass")
    
    print(f"\nChecking dataset structure:")
    print(f"  Cat annotations: {'✅' if os.path.exists(cat_path) else '❌'} {cat_path}")
    print(f"  Dog annotations: {'✅' if os.path.exists(dog_path) else '❌'} {dog_path}")
    print(f"  JPEG Images: {'✅' if os.path.exists(jpeg_path) else '❌'} {jpeg_path}")
    print(f"  Segmentation masks: {'✅' if os.path.exists(seg_path) else '❌'} {seg_path}")
    
    # Count files
    total_pairs = 0
    
    if os.path.exists(cat_path):
        cat_images = glob.glob(os.path.join(cat_path, "*.jpg"))
        cat_jsons = glob.glob(os.path.join(cat_path, "*.json"))
        cat_pairs = 0
        
        for img_file in cat_images:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            json_file = os.path.join(cat_path, f"{base_name}.json")
            if os.path.exists(json_file):
                cat_pairs += 1
        
        print(f"  Cat images: {len(cat_images)}, JSON files: {len(cat_jsons)}, Valid pairs: {cat_pairs}")
        total_pairs += cat_pairs
    
    if os.path.exists(dog_path):
        dog_images = glob.glob(os.path.join(dog_path, "*.jpg"))
        dog_jsons = glob.glob(os.path.join(dog_path, "*.json"))
        dog_pairs = 0
        
        for img_file in dog_images:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            json_file = os.path.join(dog_path, f"{base_name}.json")
            if os.path.exists(json_file):
                dog_pairs += 1
        
        print(f"  Dog images: {len(dog_images)}, JSON files: {len(dog_jsons)}, Valid pairs: {dog_pairs}")
        total_pairs += dog_pairs
    
    if os.path.exists(jpeg_path) and os.path.exists(seg_path):
        jpeg_files = glob.glob(os.path.join(jpeg_path, "*.jpg"))
        mask_files = glob.glob(os.path.join(seg_path, "*.png"))
        voc_pairs = 0
        
        for img_file in jpeg_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            mask_file = os.path.join(seg_path, f"{base_name}.png")
            if os.path.exists(mask_file):
                voc_pairs += 1
        
        print(f"  VOC JPEG images: {len(jpeg_files)}, Mask files: {len(mask_files)}, Valid pairs: {voc_pairs}")
        total_pairs += voc_pairs
    
    print(f"\n✅ Total valid image-annotation pairs: {total_pairs}")
    
    if total_pairs == 0:
        print("❌ No valid pairs found! Check dataset structure.")
        return False
    
    # Test loading a single image and annotation
    print("\nTesting single file loading...")
    
    try:
        if os.path.exists(cat_path):
            cat_images = glob.glob(os.path.join(cat_path, "*.jpg"))
            if cat_images:
                test_img = cat_images[0]
                base_name = os.path.splitext(os.path.basename(test_img))[0]
                test_json = os.path.join(cat_path, f"{base_name}.json")
                
                if os.path.exists(test_json):
                    print(f"  Testing: {os.path.basename(test_img)}")
                    
                    # Load image
                    image = cv2.imread(test_img)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"  Image shape: {image.shape}")
                    
                    # Load annotation
                    with open(test_json, 'r') as f:
                        annotation = json.load(f)
                    
                    shapes = annotation.get('shapes', [])
                    print(f"  Annotation shapes: {len(shapes)}")
                    
                    if shapes:
                        for i, shape in enumerate(shapes):
                            label = shape.get('label', 'unknown')
                            points = len(shape.get('points', []))
                            print(f"    Shape {i+1}: {label} ({points} points)")
                    
                    print("✅ Successfully loaded test image and annotation")
                    return True
        
        print("❌ Could not find test files")
        return False
        
    except Exception as e:
        print(f"❌ Error testing file loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test basic image processing without TensorFlow"""
    print("\n" + "=" * 30)
    print("Testing Image Processing")
    print("=" * 30)
    
    dataset_path = "./dataset/cat_and_dog_dataset"
    cat_path = os.path.join(dataset_path, "cats/cat")
    
    try:
        if not os.path.exists(cat_path):
            print("❌ No cat images found for processing test")
            return False
            
        cat_images = glob.glob(os.path.join(cat_path, "*.jpg"))
        if not cat_images:
            print("❌ No cat image files found")
            return False
        
        test_img = cat_images[0]
        print(f"Testing with: {os.path.basename(test_img)}")
        
        # Test OpenCV loading
        image = cv2.imread(test_img)
        if image is None:
            print("❌ Failed to load with OpenCV")
            return False
        
        print(f"✅ OpenCV loaded: {image.shape}")
        
        # Test PIL loading
        pil_image = Image.open(test_img)
        print(f"✅ PIL loaded: {pil_image.size}")
        
        # Test resizing
        resized = cv2.resize(image, (128, 128))
        print(f"✅ Resized: {resized.shape}")
        
        # Test normalization
        normalized = resized.astype(np.float32) / 255.0
        print(f"✅ Normalized: min={normalized.min():.3f}, max={normalized.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all simple tests"""
    print("Running simple dataset tests (no TensorFlow)")
    
    success1 = test_dataset_structure()
    success2 = test_image_processing()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ ALL SIMPLE TESTS PASSED!")
        print("Your dataset structure looks good.")
        print("Next: Try running 'python test_local_pipeline.py' for full TensorFlow test")
    else:
        print("❌ Some tests failed. Check dataset structure.")
    print("=" * 50)
    
    return success1 and success2

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)