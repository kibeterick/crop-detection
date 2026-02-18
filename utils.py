# -*- coding: utf-8 -*-
"""
Utility functions for Crop Disease Detection System
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing utilities"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Apply image enhancement techniques"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        try:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except Exception as e:
            logger.error(f"Noise removal failed: {e}")
            return image
    
    @staticmethod
    def auto_adjust_brightness(image: np.ndarray) -> np.ndarray:
        """Automatically adjust image brightness"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Calculate mean brightness
            mean_brightness = np.mean(v)
            
            # Adjust if too dark or too bright
            if mean_brightness < 100:
                v = cv2.add(v, int(100 - mean_brightness))
            elif mean_brightness > 180:
                v = cv2.subtract(v, int(mean_brightness - 180))
            
            # Merge and convert back
            hsv = cv2.merge([h, s, v])
            adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return adjusted
        except Exception as e:
            logger.error(f"Brightness adjustment failed: {e}")
            return image


class HistoryManager:
    """Manage detection history with database-like features"""
    
    def __init__(self, history_file: str = 'detection_history.json', max_records: int = 100):
        self.history_file = history_file
        self.max_records = max_records
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
        return []
    
    def save_history(self):
        """Save history to file"""
        try:
            # Keep only the most recent records
            if len(self.history) > self.max_records:
                self.history = self.history[-self.max_records:]
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def add_record(self, disease: str, confidence: float, features: Dict, treatment: str, image_path: str = None):
        """Add a new detection record"""
        record = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().isoformat(),
            'disease': disease,
            'confidence': confidence,
            'features': features,
            'treatment': treatment,
            'image_path': image_path
        }
        self.history.append(record)
        self.save_history()
        return record
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from history"""
        if not self.history:
            return {}
        
        diseases = [r['disease'] for r in self.history]
        confidences = [r['confidence'] for r in self.history]
        
        stats = {
            'total_detections': len(self.history),
            'unique_diseases': len(set(diseases)),
            'avg_confidence': np.mean(confidences),
            'most_common_disease': max(set(diseases), key=diseases.count) if diseases else None,
            'healthy_count': sum(1 for d in diseases if 'healthy' in d.lower()),
            'diseased_count': sum(1 for d in diseases if 'healthy' not in d.lower())
        }
        
        return stats
    
    def search(self, disease: str = None, min_confidence: float = None, 
               start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Search history with filters"""
        results = self.history.copy()
        
        if disease:
            results = [r for r in results if disease.lower() in r['disease'].lower()]
        
        if min_confidence is not None:
            results = [r for r in results if r['confidence'] >= min_confidence]
        
        if start_date:
            results = [r for r in results if r['timestamp'] >= start_date]
        
        if end_date:
            results = [r for r in results if r['timestamp'] <= end_date]
        
        return results
    
    def export_to_csv(self, filename: str = 'detection_history.csv'):
        """Export history to CSV"""
        try:
            import pandas as pd
            
            if not self.history:
                logger.warning("No history to export")
                return False
            
            # Flatten the data
            data = []
            for record in self.history:
                flat_record = {
                    'ID': record['id'],
                    'Timestamp': record['timestamp'],
                    'Disease': record['disease'],
                    'Confidence': record['confidence'],
                    'Treatment': record['treatment']
                }
                # Add features
                if 'features' in record:
                    for key, value in record['features'].items():
                        flat_record[f'Feature_{key}'] = value
                
                data.append(flat_record)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"History exported to {filename}")
            return True
        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas")
            return False
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False


class ReportGenerator:
    """Generate PDF reports for detections"""
    
    @staticmethod
    def generate_pdf_report(detection_data: Dict[str, Any], output_file: str = 'detection_report.pdf'):
        """Generate a PDF report for a detection"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
            from reportlab.lib import colors
            
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#4CAF50'),
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("🌾 Crop Disease Detection Report", title_style))
            story.append(Spacer(1, 0.3 * inch))
            
            # Detection info
            story.append(Paragraph(f"<b>Detection Date:</b> {detection_data.get('timestamp', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(f"<b>Detected Disease:</b> {detection_data.get('disease', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(f"<b>Confidence Level:</b> {detection_data.get('confidence', 0):.1f}%", styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
            
            # Features table
            if 'features' in detection_data and detection_data['features']:
                story.append(Paragraph("<b>Leaf Features:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1 * inch))
                
                feature_data = [['Feature', 'Value']]
                for key, value in detection_data['features'].items():
                    feature_data.append([key, str(value)])
                
                feature_table = Table(feature_data, colWidths=[3 * inch, 3 * inch])
                feature_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(feature_table)
                story.append(Spacer(1, 0.3 * inch))
            
            # Treatment
            story.append(Paragraph("<b>Treatment Recommendations:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(detection_data.get('treatment', 'N/A'), styles['Normal']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {output_file}")
            return True
            
        except ImportError:
            logger.error("reportlab not installed. Run: pip install reportlab")
            return False
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False


def validate_image(image_path: str) -> bool:
    """Validate if an image file is valid"""
    try:
        if not os.path.exists(image_path):
            return False
        
        # Check file size
        if os.path.getsize(image_path) < 100:  # Less than 100 bytes
            return False
        
        # Try to read the image
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Check dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        
        return True
    except Exception:
        return False


def format_confidence(confidence: float) -> str:
    """Format confidence value with color coding"""
    if confidence >= 90:
        return f"🟢 {confidence:.1f}% (High)"
    elif confidence >= 70:
        return f"🟡 {confidence:.1f}% (Medium)"
    else:
        return f"🔴 {confidence:.1f}% (Low)"
