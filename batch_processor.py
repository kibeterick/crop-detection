# -*- coding: utf-8 -*-
"""
Batch Image Processing for Crop Disease Detection
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import cv2
from tensorflow.keras import models
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from utils import ImagePreprocessor, validate_image

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple images in batch"""
    
    def __init__(self, model_path: str, class_names: List[str], max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to the trained model
            class_names: List of class names
            max_workers: Maximum number of parallel workers
        """
        self.model_path = model_path
        self.class_names = class_names
        self.max_workers = max_workers
        self.model = None
        self.results = []
        self.preprocessor = ImagePreprocessor()
        
        self.load_model()
    
    def load_model(self):
        """Load the model"""
        try:
            self.model = models.load_model(self.model_path)
            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_image(self, image_path: str, enhance: bool = True) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image_path: Path to the image
            enhance: Whether to apply image enhancement
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Validate image
            if not validate_image(image_path):
                return {
                    'image_path': image_path,
                    'success': False,
                    'error': 'Invalid image file'
                }
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'image_path': image_path,
                    'success': False,
                    'error': 'Failed to read image'
                }
            
            # Enhance if requested
            if enhance:
                image = self.preprocessor.enhance_image(image)
                image = self.preprocessor.auto_adjust_brightness(image)
            
            # Preprocess for model
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_image, Config.IMG_SIZE)
            normalized = resized.astype(np.float32) / 255.0
            input_array = np.expand_dims(normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(input_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx] * 100)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3 = [
                {
                    'disease': self.class_names[i],
                    'confidence': float(predictions[0][i] * 100)
                }
                for i in top_3_idx
            ]
            
            disease = self.class_names[predicted_idx]
            
            return {
                'image_path': image_path,
                'success': True,
                'disease': disease,
                'confidence': confidence,
                'top_3_predictions': top_3,
                'timestamp': datetime.now().isoformat(),
                'image_size': f"{image.shape[1]}x{image.shape[0]}"
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e)
            }
    
    def process_folder(self, folder_path: str, enhance: bool = True, 
                      extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Dict[str, Any]]:
        """
        Process all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            enhance: Whether to apply image enhancement
            extensions: Tuple of valid image extensions
            
        Returns:
            List of detection results
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            logger.warning(f"No images found in {folder_path}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        self.results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_image, img_path, enhance): img_path 
                for img_path in image_files
            }
            
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                
                if result['success']:
                    logger.info(f"Processed: {result['image_path']} - {result['disease']} ({result['confidence']:.1f}%)")
                else:
                    logger.error(f"Failed: {result['image_path']} - {result.get('error', 'Unknown error')}")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of batch processing"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        if not successful:
            return {
                'total': len(self.results),
                'successful': 0,
                'failed': len(failed)
            }
        
        diseases = [r['disease'] for r in successful]
        confidences = [r['confidence'] for r in successful]
        
        summary = {
            'total': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'unique_diseases': len(set(diseases)),
            'disease_distribution': {
                disease: diseases.count(disease) 
                for disease in set(diseases)
            },
            'healthy_count': sum(1 for d in diseases if 'healthy' in d.lower()),
            'diseased_count': sum(1 for d in diseases if 'healthy' not in d.lower())
        }
        
        return summary
    
    def export_results(self, output_file: str = 'batch_results.csv'):
        """Export results to CSV"""
        try:
            import pandas as pd
            
            if not self.results:
                logger.warning("No results to export")
                return False
            
            # Flatten results
            data = []
            for result in self.results:
                if result['success']:
                    row = {
                        'Image Path': result['image_path'],
                        'Disease': result['disease'],
                        'Confidence': result['confidence'],
                        'Timestamp': result['timestamp'],
                        'Image Size': result['image_size']
                    }
                    # Add top 3 predictions
                    for i, pred in enumerate(result.get('top_3_predictions', []), 1):
                        row[f'Top{i}_Disease'] = pred['disease']
                        row[f'Top{i}_Confidence'] = pred['confidence']
                else:
                    row = {
                        'Image Path': result['image_path'],
                        'Error': result.get('error', 'Unknown error')
                    }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            logger.info(f"Results exported to {output_file}")
            return True
            
        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas")
            return False
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    from main import DEMO_CLASSES
    
    processor = BatchProcessor(
        model_path='demo_model.keras',
        class_names=DEMO_CLASSES,
        max_workers=4
    )
    
    # Process a folder
    results = processor.process_folder('data/test', enhance=True)
    
    # Get summary
    summary = processor.get_summary()
    print("\n=== Batch Processing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export results
    processor.export_results('batch_results.csv')
