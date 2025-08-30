#!/usr/bin/env python3
"""
Test segmentation model with various evaluation methods
"""

import os
import cv2
import numpy as np
import json
import glob
import random

def load_model_predictions(model_path=None):
    """
    Load trained model and make predictions
    Note: This is a placeholder - you'll need TensorFlow installed
    """
    print("Note: This requires TensorFlow to be installed")
    print("For now, this will simulate predictions for demonstration")
    
    # Simulate predictions (replace with actual model loading when TF is available)
    def predict_mask(image):
        # Simple threshold-based prediction for demo
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
        return binary
    
    return predict_mask

def calculate_iou(pred_mask, true_mask):
    """
    Calculate Intersection over Union (IoU) for binary masks
    """
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    
    if np.sum(union) == 0:
        return 1.0  # Both masks are empty
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_score(pred_mask, true_mask):
    """
    Calculate Dice coefficient
    """
    intersection = np.logical_and(pred_mask, true_mask)
    dice = (2.0 * np.sum(intersection)) / (np.sum(pred_mask) + np.sum(true_mask))
    return dice

def calculate_pixel_accuracy(pred_mask, true_mask):
    """
    Calculate pixel-wise accuracy
    """
    correct = np.sum(pred_mask == true_mask)
    total = pred_mask.size
    return correct / total

def evaluate_model_on_dataset(predict_fn, test_pairs, img_size=128):
    """
    Evaluate model on test dataset
    """
    print(f"Evaluating on {len(test_pairs)} test samples...")
    
    ious = []
    dice_scores = []
    pixel_accuracies = []
    
    for i, (img_path, mask_path) in enumerate(test_pairs):
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        
        # Load ground truth mask
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        true_mask = (true_mask > 0).astype(np.uint8)  # Convert to binary
        
        # Predict mask
        pred_mask = predict_fn(image)
        pred_mask = (pred_mask > 0).astype(np.uint8)  # Ensure binary
        
        # Calculate metrics
        iou = calculate_iou(pred_mask, true_mask)
        dice = calculate_dice_score(pred_mask, true_mask)
        pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)
        
        ious.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_pairs)} samples")
    
    # Calculate mean metrics
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    
    print("\n=== Model Evaluation Results ===")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'mean_pixel_accuracy': mean_pixel_acc,
        'individual_ious': ious,
        'individual_dice': dice_scores,
        'individual_pixel_accuracies': pixel_accuracies
    }

def visualize_predictions(predict_fn, test_pairs, num_samples=5, img_size=128):
    """
    Visualize model predictions
    """
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    # Create output directory
    os.makedirs('test_results', exist_ok=True)
    
    sample_pairs = random.sample(test_pairs, min(num_samples, len(test_pairs)))
    
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (img_size, img_size))
        
        # Load ground truth
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Predict
        pred_mask = predict_fn(image_resized)
        
        # Calculate metrics for this sample
        iou = calculate_iou(pred_mask > 0, true_mask > 0)
        dice = calculate_dice_score(pred_mask > 0, true_mask > 0)
        
        # Create visualization
        fig_width = img_size * 4
        fig_height = img_size
        result_img = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        # Original image
        result_img[:img_size, :img_size] = image_resized
        
        # Ground truth (green)
        gt_colored = np.zeros_like(image_resized)
        gt_colored[:, :, 1] = true_mask  # Green channel
        result_img[:img_size, img_size:img_size*2] = gt_colored
        
        # Prediction (red)
        pred_colored = np.zeros_like(image_resized)
        pred_colored[:, :, 0] = pred_mask * 255  # Red channel
        result_img[:img_size, img_size*2:img_size*3] = pred_colored
        
        # Overlay
        overlay = cv2.addWeighted(image_resized, 0.7, pred_colored, 0.3, 0)
        result_img[:img_size, img_size*3:] = overlay
        
        # Save result
        filename = f"test_results/prediction_{i+1}_iou{iou:.3f}_dice{dice:.3f}.png"
        cv2.imwrite(filename, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        print(f"✓ Saved: {filename} (IoU: {iou:.3f}, Dice: {dice:.3f})")

def collect_test_data():
    """
    Collect test data pairs
    """
    DATASET_BASE = 'dataset/cat_and_dog_dataset'
    CATS_DIR = os.path.join(DATASET_BASE, 'cats')
    DOGS_DIR = os.path.join(DATASET_BASE, 'dogs')
    ENCODED_MASKS_DIR = os.path.join(DATASET_BASE, 'encoded_masks')
    
    # Get all images
    cat_images = glob.glob(os.path.join(CATS_DIR, '*.jpg'))
    cat_images.extend(glob.glob(os.path.join(CATS_DIR, 'cat', '*.jpg')))
    
    dog_images = glob.glob(os.path.join(DOGS_DIR, '*.jpg'))
    dog_images.extend(glob.glob(os.path.join(DOGS_DIR, 'dog', '*.jpg')))
    
    all_images = cat_images + dog_images
    
    # Match with masks
    test_pairs = []
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(ENCODED_MASKS_DIR, base_name + '.png')
        
        if os.path.exists(mask_path):
            test_pairs.append((img_path, mask_path))
    
    return test_pairs

def run_model_tests():
    """
    Run comprehensive model testing
    """
    print("=== Segmentation Model Testing ===")
    
    # Collect test data
    test_pairs = collect_test_data()
    print(f"Found {len(test_pairs)} test pairs")
    
    if len(test_pairs) == 0:
        print("No test data found!")
        return
    
    # Load model (simulated for now)
    predict_fn = load_model_predictions()
    
    # Use a subset for testing
    test_subset = random.sample(test_pairs, min(50, len(test_pairs)))
    
    # Evaluate model
    metrics = evaluate_model_on_dataset(predict_fn, test_subset)
    
    # Generate visualizations
    visualize_predictions(predict_fn, test_subset, num_samples=5)
    
    # Save results
    results = {
        'test_samples': len(test_subset),
        'metrics': metrics,
        'model_info': {
            'type': 'simulated_model',
            'description': 'This is a placeholder. Replace with actual trained model.'
        }
    }
    
    with open('test_results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Testing completed!")
    print(f"Results saved to: test_results/")
    print(f"- evaluation_results.json")
    print(f"- prediction_*.png (visualization images)")

if __name__ == "__main__":
    run_model_tests()