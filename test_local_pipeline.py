#!/usr/bin/env python3

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
sys.path.append('.')

# Fix threading issues on macOS
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from local_data_pipeline import LocalPetDataPipeline

def main():
    """Test the local data pipeline with your dataset"""
    print("=" * 50)
    print("Testing Local Dataset Pipeline")
    print("=" * 50)
    
    # Test with your dataset
    dataset_path = "./dataset/cat_and_dog_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please make sure the dataset directory exists.")
        return False
    
    print(f"✅ Dataset path found: {dataset_path}")
    
    try:
        # Initialize pipeline
        pipeline = LocalPetDataPipeline(
            dataset_path=dataset_path,
            img_size=(128, 128),
            batch_size=4
        )
        
        print("\n1. Loading image-mask pairs...")
        pairs = pipeline.load_image_and_mask_pairs()
        
        if not pairs:
            print("❌ No image-mask pairs found!")
            return False
        
        print(f"✅ Found {len(pairs)} image-annotation pairs")
        
        print("\n2. Processing and creating datasets...")
        train_ds, val_ds, test_ds, info = pipeline.load_dataset(
            train_split=0.7,
            val_split=0.2
        )
        
        print(f"✅ Dataset splits created:")
        print(f"   Train: {info['splits']['train']} samples")
        print(f"   Validation: {info['splits']['validation']} samples") 
        print(f"   Test: {info['splits']['test']} samples")
        print(f"   Total: {info['total_samples']} samples")
        
        print("\n3. Testing data iteration...")
        sample_count = 0
        for images, masks in train_ds.take(1):
            sample_count = len(images)
            print(f"✅ Successfully loaded batch with {sample_count} samples")
            print(f"   Image shape: {images.shape}")
            print(f"   Mask shape: {masks.shape}")
            print(f"   Image min/max: {images.numpy().min():.3f} / {images.numpy().max():.3f}")
            print(f"   Mask unique values: {sorted(set(masks.numpy().flatten()))}")
        
        print("\n4. Creating sample visualization...")
        pipeline.display_sample(train_ds, num_samples=min(4, sample_count))
        
        print("\n" + "=" * 50)
        print("✅ LOCAL PIPELINE TEST SUCCESSFUL!")
        print("=" * 50)
        print(f"Your dataset is ready for training with {info['total_samples']} samples")
        print("Next steps:")
        print("1. Run: python train.py (to start training)")
        print("2. Check 'local_sample_data.png' to see sample images")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)