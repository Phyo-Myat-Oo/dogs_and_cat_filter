import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import cv2

class SegmentationMetrics:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.class_names = ['Background', 'Pet', 'Border']
    
    def pixel_accuracy(self, y_true, y_pred):
        """Calculate pixel accuracy"""
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, y_pred_flat), tf.float32))
        total_pixels = tf.cast(tf.size(y_true_flat), tf.float32)
        
        return correct_pixels / total_pixels
    
    def mean_iou(self, y_true, y_pred):
        """Calculate mean Intersection over Union (IoU)"""
        ious = []
        
        for class_id in range(self.num_classes):
            # True positives, false positives, false negatives
            y_true_class = tf.cast(tf.equal(y_true, class_id), tf.float32)
            y_pred_class = tf.cast(tf.equal(y_pred, class_id), tf.float32)
            
            intersection = tf.reduce_sum(y_true_class * y_pred_class)
            union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
            
            # Avoid division by zero
            iou = tf.cond(
                tf.equal(union, 0),
                lambda: tf.constant(1.0),
                lambda: intersection / union
            )
            ious.append(iou)
        
        return tf.reduce_mean(ious)
    
    def dice_coefficient(self, y_true, y_pred, smooth=1.0):
        """Calculate Dice coefficient"""
        dice_scores = []
        
        for class_id in range(self.num_classes):
            y_true_class = tf.cast(tf.equal(y_true, class_id), tf.float32)
            y_pred_class = tf.cast(tf.equal(y_pred, class_id), tf.float32)
            
            intersection = tf.reduce_sum(y_true_class * y_pred_class)
            dice = (2 * intersection + smooth) / (tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) + smooth)
            dice_scores.append(dice)
        
        return tf.reduce_mean(dice_scores)
    
    def confusion_matrix_plot(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Generate and save confusion matrix plot"""
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(self.num_classes)))
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def classification_report_text(self, y_true, y_pred):
        """Generate classification report"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        return classification_report(
            y_true_flat, y_pred_flat,
            target_names=self.class_names,
            labels=list(range(self.num_classes))
        )

class VisualizationUtils:
    @staticmethod
    def overlay_mask_on_image(image, mask, alpha=0.5):
        """Overlay segmentation mask on original image"""
        if len(image.shape) == 3 and image.shape[-1] == 3:
            # RGB image
            overlay = image.copy()
        else:
            # Grayscale to RGB
            overlay = np.stack([image] * 3, axis=-1)
        
        # Create colored mask
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask == 1] = [255, 0, 0]  # Red for pets
        colored_mask[mask == 2] = [0, 255, 0]  # Green for borders
        
        # Blend images
        result = cv2.addWeighted(overlay.astype(np.uint8), 1-alpha, colored_mask.astype(np.uint8), alpha, 0)
        
        return result
    
    @staticmethod
    def create_prediction_grid(images, true_masks, pred_masks, num_samples=4, save_path='prediction_grid.png'):
        """Create a grid showing images, true masks, and predictions"""
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
        
        for i in range(min(num_samples, len(images))):
            # Original image
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # True mask
            axes[1, i].imshow(true_masks[i], cmap='viridis', vmin=0, vmax=2)
            axes[1, i].set_title(f'True Mask {i+1}')
            axes[1, i].axis('off')
            
            # Predicted mask
            axes[2, i].imshow(pred_masks[i], cmap='viridis', vmin=0, vmax=2)
            axes[2, i].set_title(f'Predicted {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png'):
        """Plot comparison of different metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(metrics_dict['loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, metrics_dict['loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, metrics_dict['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, metrics_dict['accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, metrics_dict['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        if 'mean_io_u' in metrics_dict:
            axes[1, 0].plot(epochs, metrics_dict['mean_io_u'], 'b-', label='Training IoU')
            axes[1, 0].plot(epochs, metrics_dict['val_mean_io_u'], 'r-', label='Validation IoU')
            axes[1, 0].set_title('Mean IoU')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('IoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in metrics_dict:
            axes[1, 1].plot(epochs, metrics_dict['lr'], 'g-', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class ModelEvaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.metrics = SegmentationMetrics()
    
    def evaluate_model(self, save_dir='evaluation_results'):
        """Comprehensive model evaluation"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Evaluating model on test dataset...")
        
        all_pixel_accuracies = []
        all_ious = []
        all_dice_scores = []
        
        all_true_masks = []
        all_pred_masks = []
        all_images = []
        
        # Evaluate on test dataset
        for batch_idx, (images, true_masks) in enumerate(self.test_dataset):
            predictions = self.model.predict(images, verbose=0)
            pred_masks = tf.argmax(predictions, axis=-1)
            
            # Calculate metrics for this batch
            for i in range(len(images)):
                true_mask = true_masks[i].numpy()
                pred_mask = pred_masks[i].numpy()
                image = images[i].numpy()
                
                # Store for later visualization
                all_images.append(image)
                all_true_masks.append(true_mask)
                all_pred_masks.append(pred_mask)
                
                # Calculate metrics
                pixel_acc = self.metrics.pixel_accuracy(true_mask, pred_mask).numpy()
                iou = self.metrics.mean_iou(true_mask, pred_mask).numpy()
                dice = self.metrics.dice_coefficient(true_mask, pred_mask).numpy()
                
                all_pixel_accuracies.append(pixel_acc)
                all_ious.append(iou)
                all_dice_scores.append(dice)
            
            # Limit evaluation to reasonable number of samples
            if batch_idx >= 10:  # Evaluate on ~10 batches
                break
        
        # Calculate summary statistics
        mean_pixel_acc = np.mean(all_pixel_accuracies)
        std_pixel_acc = np.std(all_pixel_accuracies)
        
        mean_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Pixel Accuracy: {mean_pixel_acc:.4f} ± {std_pixel_acc:.4f}")
        print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
        
        # Generate visualizations
        # Confusion matrix
        combined_true = np.concatenate(all_true_masks)
        combined_pred = np.concatenate(all_pred_masks)
        
        cm_path = f"{save_dir}/confusion_matrix.png"
        self.metrics.confusion_matrix_plot(combined_true, combined_pred, cm_path)
        
        # Classification report
        report = self.metrics.classification_report_text(combined_true, combined_pred)
        with open(f"{save_dir}/classification_report.txt", 'w') as f:
            f.write(report)
        
        # Prediction grid
        sample_indices = np.random.choice(len(all_images), min(8, len(all_images)), replace=False)
        sample_images = [all_images[i] for i in sample_indices]
        sample_true = [all_true_masks[i] for i in sample_indices]
        sample_pred = [all_pred_masks[i] for i in sample_indices]
        
        VisualizationUtils.create_prediction_grid(
            sample_images, sample_true, sample_pred,
            num_samples=min(8, len(sample_images)),
            save_path=f"{save_dir}/prediction_samples.png"
        )
        
        # Save metrics to file
        results = {
            'pixel_accuracy': {'mean': mean_pixel_acc, 'std': std_pixel_acc},
            'mean_iou': {'mean': mean_iou, 'std': std_iou},
            'dice_coefficient': {'mean': mean_dice, 'std': std_dice},
            'num_samples': len(all_images)
        }
        
        import json
        with open(f"{save_dir}/evaluation_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation results saved to '{save_dir}' directory")
        print(f"Files created:")
        print(f"- confusion_matrix.png")
        print(f"- classification_report.txt")
        print(f"- prediction_samples.png")
        print(f"- evaluation_metrics.json")
        
        return results

if __name__ == "__main__":
    # Example usage
    print("Segmentation utilities loaded successfully!")
    print("Available classes:")
    print("- SegmentationMetrics: Calculate various segmentation metrics")
    print("- VisualizationUtils: Create visualizations and plots")
    print("- ModelEvaluator: Comprehensive model evaluation")