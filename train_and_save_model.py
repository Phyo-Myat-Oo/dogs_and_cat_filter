#!/usr/bin/env python3
"""
Complete training script with model saving for cat and dog segmentation
"""

import os
import glob
import cv2
import numpy as np
import json
import random

# Configuration
IMG_SIZE = 128
DATASET_BASE = 'dataset/cat_and_dog_dataset'
CATS_DIR = os.path.join(DATASET_BASE, 'cats')
DOGS_DIR = os.path.join(DATASET_BASE, 'dogs')
ENCODED_MASKS_DIR = os.path.join(DATASET_BASE, 'encoded_masks')

def collect_data():
    """Collect all image-mask pairs"""
    print("Collecting data...")
    
    # Get all JPG images
    cat_images = glob.glob(os.path.join(CATS_DIR, '*.jpg'))
    cat_images.extend(glob.glob(os.path.join(CATS_DIR, 'cat', '*.jpg')))
    
    dog_images = glob.glob(os.path.join(DOGS_DIR, '*.jpg'))
    dog_images.extend(glob.glob(os.path.join(DOGS_DIR, 'dog', '*.jpg')))
    
    all_images = cat_images + dog_images
    
    # Match with masks
    matched_pairs = []
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(ENCODED_MASKS_DIR, base_name + '.png')
        
        if os.path.exists(mask_path):
            matched_pairs.append((img_path, mask_path))
    
    print(f"Found {len(matched_pairs)} image-mask pairs")
    return matched_pairs

def load_and_preprocess_data(pairs, batch_size=32):
    """Load and preprocess all data"""
    images = []
    masks = []
    
    print(f"Loading {len(pairs)} samples...")
    
    for i, (img_path, mask_path) in enumerate(pairs):
        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        
        # Load and resize mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        images.append(img)
        masks.append(mask)
        
        if (i + 1) % 50 == 0:
            print(f"Loaded {i + 1}/{len(pairs)} samples")
    
    return np.array(images), np.array(masks)

def build_simple_unet(input_shape, num_classes=2):
    """
    Build a simple U-Net model for segmentation
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
    except ImportError:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        return None
    
    # Encoder
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder path
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder path
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=20):
    """
    Train the segmentation model
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow not installed. Cannot train model.")
        return None
    
    print("Building model...")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_simple_unet(input_shape, num_classes=2)
    
    if model is None:
        return None
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    # Prepare callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print(f"Starting training for {epochs} epochs...")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def save_model_and_info(model, history, dataset_info):
    """
    Save trained model and training information
    """
    if model is None:
        print("❌ No model to save")
        return
    
    # Create models directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model in different formats
    print("Saving model...")
    
    # 1. Save as Keras format (recommended)
    model_path = 'saved_models/cat_dog_segmentation.keras'
    model.save(model_path)
    print(f"✅ Model saved as: {model_path}")
    
    # 2. Save as SavedModel format (for deployment)
    savedmodel_path = 'saved_models/cat_dog_segmentation_savedmodel'
    model.save(savedmodel_path, save_format='tf')
    print(f"✅ SavedModel saved as: {savedmodel_path}")
    
    # 3. Save weights only
    weights_path = 'saved_models/cat_dog_segmentation_weights.h5'
    model.save_weights(weights_path)
    print(f"✅ Weights saved as: {weights_path}")
    
    # Save training history
    if history is not None:
        history_data = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        with open('saved_models/training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        print("✅ Training history saved as: saved_models/training_history.json")
    
    # Save complete model info
    model_info = {
        'model_architecture': 'Simple U-Net',
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'output_classes': 2,
        'class_mapping': {0: 'background', 1: 'animal'},
        'training_epochs': len(history.history['loss']) if history else 0,
        'dataset_info': dataset_info
    }
    
    with open('saved_models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("✅ Model info saved as: saved_models/model_info.json")

def main():
    """Main training function"""
    print("=== Cat and Dog Segmentation Model Training ===")
    
    # Collect data
    pairs = collect_data()
    
    if len(pairs) < 10:
        print("❌ Not enough data pairs found!")
        return
    
    # Split data
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # For quick testing, use subset of data
    max_train = 100  # Use first 100 training samples
    max_val = 20     # Use first 20 validation samples
    
    train_subset = train_pairs[:min(max_train, len(train_pairs))]
    val_subset = val_pairs[:min(max_val, len(val_pairs))]
    
    print(f"Using {len(train_subset)} training and {len(val_subset)} validation samples for quick training")
    
    # Load and preprocess data
    print("Loading training data...")
    X_train, y_train = load_and_preprocess_data(train_subset)
    
    print("Loading validation data...")
    X_val, y_val = load_and_preprocess_data(val_subset)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training masks shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation masks shape: {y_val.shape}")
    
    # Dataset info
    dataset_info = {
        'total_samples': len(pairs),
        'training_samples': len(train_subset),
        'validation_samples': len(val_subset),
        'image_size': IMG_SIZE
    }
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=10)
    
    if model is not None:
        # Save everything
        save_model_and_info(model, history, dataset_info)
        
        print("\n🎉 Training completed successfully!")
        print("\nSaved files:")
        print("📁 saved_models/")
        print("  ├── cat_dog_segmentation.keras (main model file)")
        print("  ├── cat_dog_segmentation_savedmodel/ (deployment format)")
        print("  ├── cat_dog_segmentation_weights.h5 (weights only)")
        print("  ├── training_history.json (loss/accuracy curves)")
        print("  └── model_info.json (model metadata)")
        
        print("\n🧪 To test your model:")
        print("python test_trained_model.py --model_path saved_models/cat_dog_segmentation.keras --image test_image.jpg")
    
    else:
        print("❌ Training failed. Make sure TensorFlow is installed:")
        print("pip install tensorflow")

if __name__ == "__main__":
    main()