import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
import tensorflow as tf
import numpy as np
import glob
import json
from PIL import Image, ImageDraw
import cv2
from sklearn.model_selection import train_test_split

# Fix multiprocessing issues on macOS
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class LocalPetDataPipeline:
    def __init__(self, dataset_path="./dataset/cat_and_dog_dataset", img_size=(128, 128), batch_size=16):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = 3  # background, pet, border
        
        # Dataset structure paths
        self.cat_images_path = os.path.join(dataset_path, "cats/cat")
        self.dog_images_path = os.path.join(dataset_path, "dogs/dog") 
        self.jpeg_images_path = os.path.join(dataset_path, "JPEGImages")
        self.segmentation_masks_path = os.path.join(dataset_path, "SegmentationClass")
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Target image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")
        
    def load_image_and_mask_pairs(self):
        """Load image and mask pairs from the dataset"""
        image_mask_pairs = []
        
        # Check different possible structures
        if os.path.exists(self.jpeg_images_path) and os.path.exists(self.segmentation_masks_path):
            # VOC-style structure
            print("Found VOC-style structure (JPEGImages + SegmentationClass)")
            image_files = glob.glob(os.path.join(self.jpeg_images_path, "*.jpg"))
            
            for img_file in image_files:
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                mask_file = os.path.join(self.segmentation_masks_path, f"{base_name}.png")
                
                if os.path.exists(mask_file):
                    image_mask_pairs.append((img_file, mask_file))
                    
        else:
            # LabelMe-style structure with JSON annotations
            print("Found LabelMe-style structure with JSON annotations")
            
            # Process cat images
            if os.path.exists(self.cat_images_path):
                cat_images = glob.glob(os.path.join(self.cat_images_path, "*.jpg"))
                for img_file in cat_images:
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    json_file = os.path.join(self.cat_images_path, f"{base_name}.json")
                    
                    if os.path.exists(json_file):
                        image_mask_pairs.append((img_file, json_file, 'cat'))
            
            # Process dog images  
            if os.path.exists(self.dog_images_path):
                dog_images = glob.glob(os.path.join(self.dog_images_path, "*.jpg"))
                for img_file in dog_images:
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    json_file = os.path.join(self.dog_images_path, f"{base_name}.json")
                    
                    if os.path.exists(json_file):
                        image_mask_pairs.append((img_file, json_file, 'dog'))
        
        print(f"Found {len(image_mask_pairs)} image-annotation pairs")
        return image_mask_pairs
    
    def create_mask_from_labelme_json(self, json_file, img_shape, class_type='pet'):
        """Create segmentation mask from LabelMe JSON annotation"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Create blank mask
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            
            # Process shapes in the annotation
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                points = shape.get('points', [])
                
                if not points:
                    continue
                
                # Convert points to integer coordinates
                points = np.array(points, dtype=np.int32)
                
                # Determine mask value based on label
                if label in ['cat', 'dog', 'pet', 'animal']:
                    mask_value = 1  # Pet class
                elif label in ['border', 'boundary', 'edge']:
                    mask_value = 2  # Border class
                else:
                    mask_value = 1  # Default to pet class
                
                # Fill polygon
                cv2.fillPoly(mask, [points], mask_value)
            
            return mask
            
        except Exception as e:
            print(f"Error creating mask from {json_file}: {e}")
            # Return default mask with pet in center
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            h, w = img_shape[:2]
            cv2.ellipse(mask, (w//2, h//2), (w//3, h//3), 0, 0, 360, 1, -1)
            return mask
    
    def load_and_preprocess_data(self):
        """Load all data and create train/validation splits"""
        image_mask_pairs = self.load_image_and_mask_pairs()
        
        if not image_mask_pairs:
            raise ValueError("No valid image-mask pairs found in dataset")
        
        images = []
        masks = []
        
        print("Processing images and creating masks...")
        
        for i, pair in enumerate(image_mask_pairs):
            try:
                if len(pair) == 2:
                    # VOC-style: (image_file, mask_file)
                    img_file, mask_file = pair
                    
                    # Load image
                    image = cv2.imread(img_file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Load mask
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    
                elif len(pair) == 3:
                    # LabelMe-style: (image_file, json_file, class_type)
                    img_file, json_file, class_type = pair
                    
                    # Load image
                    image = cv2.imread(img_file)
                    if image is None:
                        print(f"Could not load image: {img_file}")
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Create mask from JSON
                    mask = self.create_mask_from_labelme_json(json_file, image.shape, class_type)
                
                else:
                    continue
                
                # Resize image and mask
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                
                # Normalize image
                image = image.astype(np.float32) / 255.0
                
                # Ensure mask values are in correct range
                mask = np.clip(mask, 0, self.num_classes - 1)
                
                images.append(image)
                masks.append(mask)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_mask_pairs)} images")
                    
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
        
        if not images:
            raise ValueError("No images were successfully processed")
        
        # Convert to numpy arrays
        images = np.array(images)
        masks = np.array(masks)
        
        print(f"Successfully loaded {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Unique mask values: {np.unique(masks)}")
        
        return images, masks
    
    def create_tf_dataset(self, images, masks, augment=False):
        """Create TensorFlow dataset from numpy arrays"""
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        
        if augment:
            # Use single-threaded processing to avoid mutex issues
            dataset = dataset.map(self.augment_data, num_parallel_calls=1)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)  # Reduced prefetch to avoid threading issues
        
        return dataset
    
    def augment_data(self, image, mask):
        """Data augmentation function"""
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Random vertical flip
        if tf.random.uniform(()) > 0.7:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, 0.1)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        # Ensure image is in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    
    def load_dataset(self, train_split=0.8, val_split=0.1):
        """Load and split dataset into train/validation/test"""
        try:
            # Load all data
            images, masks = self.load_and_preprocess_data()
            
            # Calculate split indices
            total_samples = len(images)
            train_size = int(total_samples * train_split)
            val_size = int(total_samples * val_split)
            
            # Create splits
            train_images = images[:train_size]
            train_masks = masks[:train_size]
            
            val_images = images[train_size:train_size + val_size]
            val_masks = masks[train_size:train_size + val_size]
            
            test_images = images[train_size + val_size:]
            test_masks = masks[train_size + val_size:]
            
            print(f"Dataset splits:")
            print(f"  Train: {len(train_images)} samples")
            print(f"  Validation: {len(val_images)} samples") 
            print(f"  Test: {len(test_images)} samples")
            
            # Create TensorFlow datasets
            train_ds = self.create_tf_dataset(train_images, train_masks, augment=True)
            val_ds = self.create_tf_dataset(val_images, val_masks, augment=False)
            test_ds = self.create_tf_dataset(test_images, test_masks, augment=False) if len(test_images) > 0 else None
            
            # Create info dict
            info = {
                'total_samples': total_samples,
                'num_classes': self.num_classes,
                'image_shape': self.img_size + (3,),
                'splits': {
                    'train': len(train_images),
                    'validation': len(val_images),
                    'test': len(test_images)
                }
            }
            
            return train_ds, val_ds, test_ds, info
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e
    
    def display_sample(self, dataset, num_samples=4, save_path='local_sample_data.png'):
        """Display sample images and masks"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            for images, masks in dataset.take(1):
                fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
                
                for i in range(min(num_samples, len(images))):
                    # Original image
                    axes[0, i].imshow(images[i])
                    axes[0, i].set_title(f'Image {i+1}')
                    axes[0, i].axis('off')
                    
                    # Mask
                    axes[1, i].imshow(masks[i], cmap='viridis', vmin=0, vmax=self.num_classes-1)
                    axes[1, i].set_title(f'Mask {i+1}')
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                plt.clf()  # Clear figure to free memory
                
                print(f"Sample images and masks saved as '{save_path}'")
                break
                
        except Exception as e:
            print(f"Error displaying samples: {e}")
            import traceback
            traceback.print_exc()

def test_local_pipeline():
    """Test the local data pipeline"""
    print("Testing Local Pet Data Pipeline...")
    
    # Initialize pipeline
    pipeline = LocalPetDataPipeline(
        dataset_path="./dataset/cat_and_dog_dataset",
        img_size=(128, 128),
        batch_size=4
    )
    
    try:
        # Load dataset
        train_ds, val_ds, test_ds, info = pipeline.load_dataset()
        
        print("Dataset loaded successfully!")
        print(f"Info: {info}")
        
        # Display samples
        pipeline.display_sample(train_ds, num_samples=4)
        
        print("Local data pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    test_local_pipeline()