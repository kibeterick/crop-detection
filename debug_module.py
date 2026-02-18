#!/usr/bin/env python3
import leaf_detector
import sys

print("Module path:", leaf_detector.__file__)
print("Module contents:", dir(leaf_detector))

# Try to execute the file manually
print("\nTrying to execute file content...")
try:
    with open('leaf_detector.py', 'r') as f:
        content = f.read()
    
    # Create a new namespace
    namespace = {}
    exec(content, namespace)
    
    print("Execution successful!")
    print("Namespace contents:", [k for k in namespace.keys() if not k.startswith('__')])
    
    if 'LeafDetector' in namespace:
        print("✓ LeafDetector found in namespace")
    else:
        print("✗ LeafDetector NOT found in namespace")
        
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()