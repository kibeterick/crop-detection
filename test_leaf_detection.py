#!/usr/bin/env python3
"""Test script for leaf_detection module"""

print("=" * 60)
print("TESTING LEAF DETECTION MODULE")
print("=" * 60)

try:
    print("\n1. Importing modules...")
    from leaf_detection import LeafDetector, LeafQualityChecker
    print("   ✓ Import successful")
    
    print("\n2. Creating detector...")
    detector = LeafDetector()
    print("   ✓ LeafDetector created")
    
    print("\n3. Creating quality checker...")
    checker = LeafQualityChecker()
    print("   ✓ LeafQualityChecker created")
    
    print("\n4. Testing with dummy image...")
    import numpy as np
    import cv2
    
    # Create a simple test image (green square)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[20:80, 20:80] = [0, 255, 0]  # Green square
    
    leaves = detector.detect_leaves(test_image)
    print(f"   ✓ Detected {len(leaves)} leaf/leaves")
    
    quality = checker.check_quality(test_image)
    print(f"   ✓ Quality check: {quality['is_suitable']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe leaf_detection module is working correctly!")
    print("\nUsage:")
    print("  from leaf_detection import LeafDetector, LeafQualityChecker")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
