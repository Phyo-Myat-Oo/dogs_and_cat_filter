import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Configuration
DATA_DIR = 'dataset/cat_and_dog_dataset'
IMAGE_DIR = os.path.join(DATA_DIR, 'JPEGImages')
MASK_DIR = os.path.join(DATA_DIR, 'SegmentationClass')
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5

def find_num_classes(mask_dir):
    """Find number of unique classes in masks"""
    print(f"Scanning masks in '{mask_dir}'...")
    unique_values = set()
    mask_files = os.listdir(mask_dir)
    
    for filename in tqdm(mask_files, desc="Finding unique classes"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            unique_values.update(np.unique(mask))
    
    sorted_values = sorted(list(unique_values))
    print(f"Found {len(sorted_values)} unique classes: {sorted_values}")
    return len(sorted_values)

def load_and_preprocess(img_path, mask_path):
    """Load and preprocess image and mask"""
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    
    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    
    return img, mask

def build_unet(input_shape, num_classes):
    """Build simple U-Net model"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

def create_datasets():
    """Create train and validation datasets"""
    # Get file paths
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]
    mask_paths = [os.path.join(MASK_DIR, f) for f in image_files]
    
    # Split dataset
    dataset_size = len(image_paths)
    train_size = int(0.8 * dataset_size)
    
    # Create datasets
    full_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    full_dataset = full_dataset.shuffle(buffer_size=dataset_size)
    
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    # Preprocess datasets
    train_batches = (train_dataset
                    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                    .cache()
                    .shuffle(200)
                    .batch(BATCH_SIZE)
                    .repeat()
                    .prefetch(tf.data.AUTOTUNE))
    
    val_batches = (val_dataset
                  .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))
    
    steps_per_epoch = train_size // BATCH_SIZE
    
    print(f"Total examples: {dataset_size}")
    print(f"Training examples: {train_size}")
    print(f"Validation examples: {dataset_size - train_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    return train_batches, val_batches, steps_per_epoch

def train_model():
    """Main training function"""
    print("=== Simple U-Net Training ===")
    
    # Find number of classes
    print("1. Finding number of classes...")
    global NUM_CLASSES
    NUM_CLASSES = find_num_classes(MASK_DIR)
    
    # Create datasets
    print("2. Creating datasets...")
    train_batches, val_batches, steps_per_epoch = create_datasets()
    
    # Build model
    print("3. Building U-Net model...")
    model = build_unet((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("Model summary:")
    model.summary()
    
    # Train model
    print("4. Starting training...")
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    history = model.fit(
        train_batches,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_batches,
        callbacks=[lr_scheduler]
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/simple_unet.keras')
    print("Model saved to models/simple_unet.keras")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate model
    print("5. Evaluating model...")
    loss, accuracy = model.evaluate(val_batches)
    print(f"Final validation loss: {loss:.4f}")
    print(f"Final validation accuracy: {accuracy:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()