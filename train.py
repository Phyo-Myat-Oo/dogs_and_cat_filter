import os
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from local_data_pipeline import LocalPetDataPipeline
from segmentation_model import create_model

class SegmentationTrainer:
    def __init__(self, img_size=(128, 128), batch_size=16, epochs=50):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.history = None
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def setup_data(self):
        print("Setting up data pipeline...")
        pipeline = LocalPetDataPipeline(
            dataset_path="./dataset/cat_and_dog_dataset",
            img_size=self.img_size, 
            batch_size=self.batch_size
        )
        self.train_ds, self.val_ds, self.test_ds, self.dataset_info = pipeline.load_dataset()
        
        print("Data pipeline ready!")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Dataset info: {self.dataset_info}")
        
        return pipeline
    
    def setup_model(self, use_pretrained=True, learning_rate=1e-4):
        print("Setting up model...")
        input_shape = (*self.img_size, 3)
        self.model = create_model(
            input_shape=input_shape,
            num_classes=3,
            use_pretrained=use_pretrained,
            learning_rate=learning_rate
        )
        print("Model ready!")
        
    def setup_callbacks(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'checkpoints/best_model_{timestamp}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/{timestamp}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def train(self, use_pretrained=True, learning_rate=1e-4):
        print("Starting training process...")
        
        # Setup
        pipeline = self.setup_data()
        self.setup_model(use_pretrained, learning_rate)
        callbacks = self.setup_callbacks()
        
        # Display sample data
        pipeline.display_sample(self.train_ds)
        
        print(f"\nTraining for {self.epochs} epochs...")
        print("=" * 50)
        
        # Train the model
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'results/final_model_{timestamp}.h5'
        self.model.save(model_path)
        print(f"Final model saved to: {model_path}")
        
        return self.history
    
    def plot_training_history(self, save_path='results/training_history.png'):
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training & validation accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot IoU scores
        if 'mean_io_u' in self.history.history:
            axes[1, 0].plot(self.history.history['mean_io_u'], label='Training IoU')
            axes[1, 0].plot(self.history.history['val_mean_io_u'], label='Validation IoU')
            axes[1, 0].set_title('Mean IoU Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {save_path}")
    
    def predict_and_visualize(self, num_samples=4, save_path='results/predictions.png'):
        if self.model is None:
            print("No model available. Train the model first.")
            return
        
        # Get sample batch
        for images, true_masks in self.val_ds.take(1):
            predictions = self.model.predict(images)
            predicted_masks = tf.argmax(predictions, axis=-1)
            
            fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
            
            for i in range(min(num_samples, len(images))):
                # Original image
                axes[0, i].imshow(images[i])
                axes[0, i].set_title(f'Original Image {i+1}')
                axes[0, i].axis('off')
                
                # True mask
                axes[1, i].imshow(true_masks[i], cmap='viridis')
                axes[1, i].set_title(f'True Mask {i+1}')
                axes[1, i].axis('off')
                
                # Predicted mask
                axes[2, i].imshow(predicted_masks[i], cmap='viridis')
                axes[2, i].set_title(f'Predicted Mask {i+1}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Prediction visualization saved to: {save_path}")
            break

def main():
    # Configuration
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 8  # Reduced for local dataset
    EPOCHS = 30
    USE_PRETRAINED = True
    LEARNING_RATE = 1e-4
    
    print("=== Pet Segmentation Training ===")
    print(f"Dataset: Local cat-dog dataset (./dataset/cat_and_dog_dataset)")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Using pretrained encoder: {USE_PRETRAINED}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SegmentationTrainer(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    try:
        # Train the model
        history = trainer.train(
            use_pretrained=USE_PRETRAINED,
            learning_rate=LEARNING_RATE
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Generate predictions
        trainer.predict_and_visualize()
        
        print("\n=== Training Complete ===")
        print("Check the 'results' folder for:")
        print("- training_history.png: Training metrics plots")
        print("- predictions.png: Sample predictions")
        print("- final_model_*.h5: Saved model")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()