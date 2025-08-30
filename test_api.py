#!/usr/bin/env python3

import requests
import json
import base64
import os
import glob
from PIL import Image
import io

def test_health_endpoint(base_url="http://localhost:5001"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/")
        print("Health Check Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_model_status(base_url="http://localhost:5001"):
    """Test the model status endpoint"""
    try:
        response = requests.get(f"{base_url}/model/status")
        print("Model Status Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing model status: {e}")
        return False

def test_training_status(base_url="http://localhost:5001"):
    """Test the training status endpoint"""
    try:
        response = requests.get(f"{base_url}/train/status")
        print("Training Status Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing training status: {e}")
        return False

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_segmentation_with_file(image_path, base_url="http://localhost:5001"):
    """Test segmentation endpoint with file upload"""
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/segment", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("Segmentation successful!")
            print(f"Input shape: {result.get('input_shape')}")
            print(f"Mask shape: {result.get('mask_shape')}")
            print(f"Confidence scores: {result.get('confidence_scores')}")
            
            # Save the result mask
            if 'mask_base64' in result:
                mask_data = base64.b64decode(result['mask_base64'])
                with open('segmentation_result.png', 'wb') as f:
                    f.write(mask_data)
                print("Segmentation mask saved as 'segmentation_result.png'")
            
            return True
        else:
            print(f"Segmentation failed with status {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"Error testing segmentation: {e}")
        return False

def test_segmentation_with_base64(image_path, base_url="http://localhost:5001"):
    """Test segmentation endpoint with base64 encoded image"""
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return False
    
    try:
        image_b64 = image_to_base64(image_path)
        
        data = {
            'image_data': image_b64
        }
        
        response = requests.post(
            f"{base_url}/segment",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Base64 segmentation successful!")
            print(f"Input shape: {result.get('input_shape')}")
            print(f"Mask shape: {result.get('mask_shape')}")
            print(f"Confidence scores: {result.get('confidence_scores')}")
            return True
        else:
            print(f"Base64 segmentation failed with status {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"Error testing base64 segmentation: {e}")
        return False

def test_legacy_predict(base_url="http://localhost:5001"):
    """Test the legacy predict endpoint for backward compatibility"""
    try:
        data = {
            'data': [1, 2, 3, 4, 5]
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Legacy predict endpoint working!")
            print(f"Result: {result}")
            return True
        else:
            print(f"Legacy predict failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing legacy predict: {e}")
        return False

def create_test_image(size=(128, 128)):
    """Create a simple test image if no real images are available"""
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create a simple test image with some shapes
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes to simulate a pet-like image
    # Background
    draw.rectangle([(0, 0), size], fill='lightblue')
    
    # "Pet" shape (ellipse)
    pet_size = (size[0]//2, size[1]//2)
    pet_pos = (size[0]//4, size[1]//4)
    draw.ellipse([pet_pos, (pet_pos[0] + pet_size[0], pet_pos[1] + pet_size[1])], 
                fill='brown', outline='black', width=2)
    
    # Save test image
    test_path = 'test_image.jpg'
    img.save(test_path)
    print(f"Created test image: {test_path}")
    
    return test_path

def find_test_images():
    """Find available test images"""
    # Look for images in common locations
    image_patterns = [
        'test_*.jpg', 'test_*.png', 'test_*.jpeg',
        '*.jpg', '*.png', '*.jpeg',
        'dataset/**/*.jpg', 'dataset/**/*.png', 'dataset/**/*.jpeg',
        'sample_*.jpg', 'sample_*.png'
    ]
    
    found_images = []
    for pattern in image_patterns:
        found_images.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and limit to first few
    found_images = list(set(found_images))[:5]
    
    return found_images

def run_comprehensive_test(base_url="http://localhost:5001"):
    """Run all API tests"""
    print("="*50)
    print("COMPREHENSIVE API TESTING")
    print("="*50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    total_tests += 1
    if test_health_endpoint(base_url):
        tests_passed += 1
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")
    
    # Test 2: Model status
    print("\n2. Testing model status...")
    total_tests += 1
    if test_model_status(base_url):
        tests_passed += 1
        print("✅ Model status check passed")
    else:
        print("❌ Model status check failed")
    
    # Test 3: Training status
    print("\n3. Testing training status...")
    total_tests += 1
    if test_training_status(base_url):
        tests_passed += 1
        print("✅ Training status check passed")
    else:
        print("❌ Training status check failed")
    
    # Test 4: Legacy predict
    print("\n4. Testing legacy predict endpoint...")
    total_tests += 1
    if test_legacy_predict(base_url):
        tests_passed += 1
        print("✅ Legacy predict passed")
    else:
        print("❌ Legacy predict failed")
    
    # Test 5 & 6: Segmentation tests
    test_images = find_test_images()
    
    if not test_images:
        print("\nNo test images found, creating a synthetic test image...")
        test_images = [create_test_image()]
    
    if test_images:
        test_image = test_images[0]
        print(f"\nUsing test image: {test_image}")
        
        # Test 5: File upload segmentation
        print("\n5. Testing segmentation with file upload...")
        total_tests += 1
        if test_segmentation_with_file(test_image, base_url):
            tests_passed += 1
            print("✅ File upload segmentation passed")
        else:
            print("❌ File upload segmentation failed")
        
        # Test 6: Base64 segmentation
        print("\n6. Testing segmentation with base64...")
        total_tests += 1
        if test_segmentation_with_base64(test_image, base_url):
            tests_passed += 1
            print("✅ Base64 segmentation passed")
        else:
            print("❌ Base64 segmentation failed")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    elif tests_passed > total_tests // 2:
        print("⚠️  Most tests passed, some issues detected")
    else:
        print("❌ Multiple test failures detected")
    
    return tests_passed, total_tests

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Flask segmentation API')
    parser.add_argument('--url', type=str, default='http://localhost:5001', 
                       help='Base URL of the API')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--test', type=str, choices=['health', 'model', 'segment', 'all'],
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    if args.test == 'health':
        test_health_endpoint(args.url)
    elif args.test == 'model':
        test_model_status(args.url)
    elif args.test == 'segment':
        if args.image:
            test_segmentation_with_file(args.image, args.url)
        else:
            print("Please provide an image path with --image")
    else:
        run_comprehensive_test(args.url)

if __name__ == "__main__":
    main()