# Debug script for leaf_detector.py
import sys

print("Testing leaf_detector.py imports...")

try:
    print("1. Testing basic import...")
    import leaf_detector
    print("✓ Basic import successful")
    
    print("2. Checking module contents...")
    contents = dir(leaf_detector)
    print(f"Module contents: {contents}")
    
    print("3. Testing class imports...")
    try:
        from leaf_detector import LeafDetector
        print("✓ LeafDetector imported successfully")
    except ImportError as e:
        print(f"✗ LeafDetector import failed: {e}")
    
    try:
        from leaf_detector import LeafQualityChecker
        print("✓ LeafQualityChecker imported successfully")
    except ImportError as e:
        print(f"✗ LeafQualityChecker import failed: {e}")
        
    print("4. Testing dependencies...")
    import cv2
    import numpy as np
    import logging
    from typing import Tuple, List, Optional, Dict, Any
    print("✓ All dependencies available")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")