# -*- coding: utf-8 -*-
"""
System Test Suite for Crop Disease Detection
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages can be imported"""
    print("\n=== Testing Package Imports ===")
    
    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'tensorflow': 'TensorFlow',
        'tkinter': 'Tkinter',
        'requests': 'Requests',
        'geopy': 'Geopy (optional)',
        'pyttsx3': 'pyttsx3 (optional)',
        'pandas': 'Pandas (optional)',
        'reportlab': 'ReportLab (optional)'
    }
    
    results = {}
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
            results[name] = True
        except ImportError:
            if 'optional' in name.lower():
                print(f"⚠ {name} - Not installed (optional)")
                results[name] = 'optional'
            else:
                print(f"✗ {name} - MISSING (required)")
                results[name] = False
    
    return results


def test_tensorflow():
    """Test TensorFlow installation and GPU availability"""
    print("\n=== Testing TensorFlow ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ No GPU detected - will use CPU")
        
        # Test basic operation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0], [1.0]])
        result = tf.matmul(x, y)
        print("✓ TensorFlow operations working")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False


def test_opencv():
    """Test OpenCV installation"""
    print("\n=== Testing OpenCV ===")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("⚠ Camera opened but cannot read frames")
            cap.release()
        else:
            print("⚠ No camera detected")
        
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n=== Testing Model Creation ===")
    
    try:
        from tensorflow.keras import layers, models
        import numpy as np
        
        # Create a simple model
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(16, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✓ Model creation successful")
        
        # Test prediction
        dummy_input = np.random.random((1, 224, 224, 3))
        output = model.predict(dummy_input, verbose=0)
        print(f"✓ Model prediction working - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def test_file_structure():
    """Test if required files and directories exist"""
    print("\n=== Testing File Structure ===")
    
    required_files = [
        'main.py',
        'config.py',
        'utils.py',
        'batch_processor.py',
        'model_trainer.py',
        'requirements.txt',
        'README.md'
    ]
    
    optional_files = [
        'demo_model.keras',
        'crop_disease_cnn_model.keras'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✓ {file} (optional)")
        else:
            print(f"⚠ {file} - Not found (will be created on first run)")
    
    return all_good


def test_custom_modules():
    """Test custom modules"""
    print("\n=== Testing Custom Modules ===")
    
    try:
        from config import Config
        print("✓ config.py imported")
        
        from utils import ImagePreprocessor, HistoryManager, ReportGenerator
        print("✓ utils.py imported")
        
        from batch_processor import BatchProcessor
        print("✓ batch_processor.py imported")
        
        from model_trainer import ModelTrainer
        print("✓ model_trainer.py imported")
        
        return True
    except Exception as e:
        print(f"✗ Custom module import failed: {e}")
        return False


def test_voice():
    """Test voice assistant"""
    print("\n=== Testing Voice Assistant ===")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"✓ Voice engine initialized - {len(voices)} voice(s) available")
        return True
    except ImportError:
        print("⚠ pyttsx3 not installed - voice features will be disabled")
        return 'optional'
    except Exception as e:
        print(f"⚠ Voice test failed: {e}")
        return False


def test_geopy():
    """Test geopy for agrovet finder"""
    print("\n=== Testing Geopy ===")
    
    try:
        from geopy.geocoders import Nominatim
        from geopy.distance import geodesic
        
        geolocator = Nominatim(user_agent="test_app")
        print("✓ Geopy initialized")
        
        # Test geocoding
        location = geolocator.geocode("Nairobi, Kenya", timeout=5)
        if location:
            print(f"✓ Geocoding working - Nairobi: ({location.latitude:.4f}, {location.longitude:.4f})")
        else:
            print("⚠ Geocoding test failed")
        
        return True
    except ImportError:
        print("⚠ Geopy not installed - agrovet finder will be limited")
        return 'optional'
    except Exception as e:
        print(f"⚠ Geopy test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CROP DISEASE DETECTION SYSTEM - SYSTEM TEST")
    print("=" * 60)
    
    results = {}
    
    results['imports'] = test_imports()
    results['tensorflow'] = test_tensorflow()
    results['opencv'] = test_opencv()
    results['model'] = test_model_creation()
    results['files'] = test_file_structure()
    results['modules'] = test_custom_modules()
    results['voice'] = test_voice()
    results['geopy'] = test_geopy()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    optional = sum(1 for v in results.values() if v == 'optional')
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Optional (not installed): {optional}")
    
    if failed == 0:
        print("\n✓ All required tests passed! System is ready to use.")
        print("\nTo start the application, run:")
        print("  python main.py")
        return True
    else:
        print("\n✗ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
