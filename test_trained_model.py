#!/usr/bin/env python3
"""
Test a trained TensorFlow segmentation model
Usage: python test_trained_model.py --model_path your_model.keras --test_images path/to/test/images
"""

import argparse
import os
import glob
import numpy as np
import cv2
import json

def load_tensorflow_model(model_path):
    """
    Load a trained TensorFlow model
    """
    try:
        import tensorflow as tf
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        return model
    except ImportError:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_single_image(model, image_path, target_size=(128, 128)):
    """
    Predict segmentation mask for a single image
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        return None, None
        
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for model input
    resized_image = cv2.resize(original_image, target_size)
    input_image = resized_image.astype(np.float32) / 255.0
    input_batch = np.expand_dims(input_image, axis=0)
    
    # Predict
    prediction = model.predict(input_batch, verbose=0)
    
    # Convert to mask (assuming binary segmentation)
    if prediction.shape[-1] > 1:  # Multi-class
        mask = np.argmax(prediction[0], axis=-1)
    else:  # Binary
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
    
    return original_image, mask

def test_on_single_image(model_path, image_path):
    """
    Test model on a single image
    """
    print(f"=== Testing Single Image ===")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    
    # Load model
    model = load_tensorflow_model(model_path)
    if model is None:
        return
    
    # Predict
    original_image, predicted_mask = predict_single_image(model, image_path)
    
    if original_image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Save results
    os.makedirs('single_test_results', exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original
    cv2.imwrite(f'single_test_results/{base_name}_original.png', 
                cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    
    # Save mask
    cv2.imwrite(f'single_test_results/{base_name}_mask.png', predicted_mask * 255)
    
    # Save overlay
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :, 0] = predicted_mask * 255  # Red channel
    overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite(f'single_test_results/{base_name}_overlay.png', 
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Results saved to single_test_results/")
    print(f"   - {base_name}_original.png")
    print(f"   - {base_name}_mask.png") 
    print(f"   - {base_name}_overlay.png")

def test_on_directory(model_path, test_dir):
    """
    Test model on all images in a directory
    """
    print(f"=== Testing Directory ===")
    print(f"Model: {model_path}")
    print(f"Directory: {test_dir}")
    
    # Load model
    model = load_tensorflow_model(model_path)
    if model is None:
        return
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    if not image_paths:
        print(f"❌ No images found in {test_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Create results directory
    os.makedirs('batch_test_results', exist_ok=True)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        original_image, predicted_mask = predict_single_image(model, image_path)
        
        if original_image is None:
            print(f"  ❌ Failed to process {image_path}")
            continue
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save results
        mask_path = f'batch_test_results/{base_name}_mask.png'
        overlay_path = f'batch_test_results/{base_name}_overlay.png'
        
        cv2.imwrite(mask_path, predicted_mask * 255)
        
        # Create overlay
        mask_colored = np.zeros_like(original_image)
        mask_colored[:, :, 0] = predicted_mask * 255
        overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Calculate basic statistics
        animal_pixels = np.sum(predicted_mask)
        total_pixels = predicted_mask.size
        animal_ratio = animal_pixels / total_pixels
        
        results.append({
            'image': base_name,
            'animal_pixels': int(animal_pixels),
            'total_pixels': int(total_pixels),
            'animal_ratio': float(animal_ratio)
        })
        
        print(f"  ✅ Animal coverage: {animal_ratio:.1%}")
    
    # Save summary
    summary = {
        'model_path': model_path,
        'test_directory': test_dir,
        'total_images': len(image_paths),
        'processed_images': len(results),
        'results': results
    }
    
    with open('batch_test_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Batch testing completed!")
    print(f"Results saved to: batch_test_results/")
    print(f"Summary: batch_test_results/summary.json")

def main():
    parser = argparse.ArgumentParser(description='Test trained segmentation model')
    parser.add_argument('--model_path', required=True, help='Path to trained model file (.keras)')
    parser.add_argument('--image', help='Single image to test')
    parser.add_argument('--test_dir', help='Directory of images to test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        test_on_single_image(args.model_path, args.image)
    
    elif args.test_dir:
        if not os.path.isdir(args.test_dir):
            print(f"❌ Directory not found: {args.test_dir}")
            return
        test_on_directory(args.model_path, args.test_dir)
    
    else:
        print("❌ Please specify either --image or --test_dir")
        print("Examples:")
        print("  python test_trained_model.py --model_path model.keras --image test.jpg")
        print("  python test_trained_model.py --model_path model.keras --test_dir test_images/")

if __name__ == "__main__":
    main()