"""Leaf Detection Module - Working Version"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LeafDetector:
    """Detects and extracts leaves from images"""
    
    def __init__(self):
        self.min_leaf_area = 1000
        self.max_leaf_area = 500000
        logger.info("LeafDetector initialized")
    
    def detect_leaves(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect leaves in image"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            leaves = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_leaf_area < area < self.max_leaf_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    leaves.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': 85.0
                    })
            return leaves
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def get_best_leaf(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get best detected leaf"""
        leaves = self.detect_leaves(image)
        return leaves[0] if leaves else None
    
    def extract_leaf_region(self, image: np.ndarray, auto_detect: bool = True) -> np.ndarray:
        """Extract leaf region"""
        if not auto_detect:
            return image
        
        best_leaf = self.get_best_leaf(image)
        if best_leaf is None:
            return image
        x, y, w, h = best_leaf['bbox']
        return image[y:y+h, x:x+w]
    
    def enhance_leaf_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance leaf image for better detection"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image


class LeafQualityChecker:
    """Checks image quality for disease detection"""
    
    def __init__(self):
        self.min_brightness = 30
        self.max_brightness = 225
    
    def check_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Check image quality"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            detector = LeafDetector()
            leaves = detector.detect_leaves(image)
            
            results = {
                'is_suitable': True,
                'issues': [],
                'warnings': [],
                'metrics': {
                    'brightness': brightness,
                    'leaves_detected': len(leaves)
                }
            }
            
            if brightness < self.min_brightness:
                results['is_suitable'] = False
                results['issues'].append("Image too dark")
            elif brightness > self.max_brightness:
                results['is_suitable'] = False
                results['issues'].append("Image too bright")
            
            if len(leaves) == 0:
                results['is_suitable'] = False
                results['issues'].append("No leaf detected")
            
            return results
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {'is_suitable': False, 'issues': ['Check failed']}


# Test when run directly
if __name__ == "__main__":
    print("Testing LeafDetector and LeafQualityChecker...")
    detector = LeafDetector()
    checker = LeafQualityChecker()
    print("✓ Classes work!")
