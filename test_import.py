#!/usr/bin/env python3
# Test import script

print("Starting import test...")

try:
    print("1. Testing basic import...")
    import leaf_detector
    print("✓ Module imported")
    
    print("2. Testing class import...")
    from leaf_detector import LeafDetector
    print("✓ LeafDetector imported")
    
    from leaf_detector import LeafQualityChecker
    print("✓ LeafQualityChecker imported")
    
    print("3. Testing class creation...")
    detector = LeafDetector()
    checker = LeafQualityChecker()
    print("✓ Classes created")
    
    print("\n✅ ALL TESTS PASSED!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()