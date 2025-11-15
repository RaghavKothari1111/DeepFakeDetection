#!/usr/bin/env python3
"""
Test script to diagnose model loading and prediction issues.
"""
import os
import sys
import traceback
import numpy as np
from PIL import Image

# Add app directory to path
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, os.path.join(backend_dir, 'app'))

from models_interface import MODEL_REGISTRY, _load_model_entry, _preprocess_for_keras, _preprocess_for_torch, _predict_with_keras, _predict_with_torch, TORCH_AVAILABLE, TF_AVAILABLE

def create_test_image(size=(224, 224)):
    """Create a simple test image"""
    img = Image.new('RGB', size, color='red')
    return img

def test_model(entry):
    """Test loading and prediction for a single model"""
    name = entry.get("name", "unknown")
    print(f"\n{'='*60}")
    print(f"Testing model: {name}")
    print(f"{'='*60}")
    
    # Test 1: Model loading
    print(f"\n[1] Testing model loading...")
    try:
        model = _load_model_entry(entry)
        print(f"✓ Model loaded successfully")
        
        # Try to get model info
        if TF_AVAILABLE and hasattr(model, 'input_shape'):
            try:
                print(f"  Input shape: {model.input_shape}")
            except:
                pass
        if TF_AVAILABLE and hasattr(model, 'summary'):
            try:
                print(f"  Model has summary method")
            except:
                pass
                
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Image preprocessing
    print(f"\n[2] Testing image preprocessing...")
    try:
        input_size = entry.get("input_size", 224)
        test_img = create_test_image((input_size, input_size))
        
        framework = entry.get("framework", "").lower()
        if "torch" in framework or entry.get("path", "").endswith((".pt", ".pth")):
            if not TORCH_AVAILABLE:
                print("✗ PyTorch not available")
                return False
            preprocessed = _preprocess_for_torch(test_img, input_size)
            print(f"✓ Preprocessing successful (PyTorch)")
            print(f"  Shape: {preprocessed.shape}")
        else:
            if not TF_AVAILABLE:
                print("✗ TensorFlow not available")
                return False
            preprocessed = _preprocess_for_keras(test_img, input_size)
            print(f"✓ Preprocessing successful (Keras)")
            print(f"  Shape: {preprocessed.shape}")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Prediction
    print(f"\n[3] Testing prediction...")
    try:
        if "torch" in framework or entry.get("path", "").endswith((".pt", ".pth")):
            probs = _predict_with_torch(model, preprocessed)
        else:
            probs = _predict_with_keras(model, preprocessed)
        
        print(f"✓ Prediction successful")
        print(f"  Output shape: {probs.shape}")
        print(f"  Output values: {probs}")
        print(f"  Sum: {probs.sum():.4f}")
        
        # Check if output is reasonable
        if probs.size >= 2:
            print(f"  [real, fake] = [{probs[0]:.4f}, {probs[1]:.4f}]")
        else:
            print(f"  Single value: {probs[0]:.4f}")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n✓ All tests passed for {name}")
    return True

if __name__ == "__main__":
    print("Model Testing Script")
    print("="*60)
    
    results = {}
    for entry in MODEL_REGISTRY:
        name = entry.get("name", "unknown")
        results[name] = test_model(entry)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} models working")

