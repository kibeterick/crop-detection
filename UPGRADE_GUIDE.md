# Upgrade Guide - Crop Disease Detection System

## Overview
This guide explains how to upgrade your existing system with the new enhancements.

## What's New

### New Files
1. **requirements.txt** - Dependency management
2. **config.py** - Centralized configuration
3. **utils.py** - Utility functions and helpers
4. **batch_processor.py** - Batch image processing
5. **model_trainer.py** - Custom model training
6. **test_system.py** - System testing suite
7. **install.bat** - Windows installation script
8. **README.md** - Comprehensive documentation
9. **ENHANCEMENTS.md** - Enhancement details
10. **UPGRADE_GUIDE.md** - This file

### Enhanced Features in main.py
- Better error handling
- Improved image preprocessing
- Enhanced agrovet finder (finds truly nearest stores)
- Better logging
- More robust model management

## Installation Steps

### Step 1: Backup Your Current System
```bash
# Create a backup folder
mkdir backup
copy *.py backup\
copy *.json backup\
copy *.h5 backup\
copy *.keras backup\
```

### Step 2: Install New Dependencies
```bash
# Run the installer (Windows)
install.bat

# Or manually install
pip install -r requirements.txt
```

### Step 3: Test the System
```bash
python test_system.py
```

This will verify:
- All packages are installed correctly
- TensorFlow is working
- Camera is accessible
- Models can be created
- All modules can be imported

### Step 4: Run the Enhanced Application
```bash
python main.py
```

## New Capabilities

### 1. Batch Processing

Process multiple images at once:

```python
from batch_processor import BatchProcessor
from main import PLANTVILLAGE_CLASSES

# Initialize
processor = BatchProcessor(
    model_path='your_model.keras',
    class_names=PLANTVILLAGE_CLASSES,
    max_workers=4  # Parallel processing
)

# Process a folder
results = processor.process_folder('data/test', enhance=True)

# Get statistics
summary = processor.get_summary()
print(f"Processed: {summary['total']} images")
print(f"Success: {summary['successful']}")
print(f"Average confidence: {summary['avg_confidence']:.1f}%")

# Export to CSV
processor.export_results('batch_results.csv')
```

### 2. Custom Model Training

Train your own models:

```python
from model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    data_dir='data/train',
    img_size=(224, 224),
    batch_size=32
)

# Build model (choose architecture)
trainer.build_model(num_classes=38, architecture='mobilenet')
# Options: 'simple', 'mobilenet', 'resnet'

# Train
history = trainer.train(epochs=20, save_path='my_model.keras')

# Visualize training
trainer.plot_training_history('training_plot.png')

# Evaluate on test set
trainer.evaluate('data/test')
```

### 3. Advanced Image Preprocessing

```python
from utils import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Enhance image
enhanced = preprocessor.enhance_image(image)

# Remove noise
clean = preprocessor.remove_noise(image)

# Auto-adjust brightness
adjusted = preprocessor.auto_adjust_brightness(image)
```

### 4. History Management

```python
from utils import HistoryManager

# Initialize
history = HistoryManager(max_records=100)

# Add detection
history.add_record(
    disease='Tomato___Late_blight',
    confidence=95.5,
    features={'color': 'RGB(120,150,80)'},
    treatment='Apply fungicide...',
    image_path='image.jpg'
)

# Get statistics
stats = history.get_statistics()
print(f"Total detections: {stats['total_detections']}")
print(f"Most common: {stats['most_common_disease']}")

# Search history
results = history.search(
    disease='Tomato',
    min_confidence=80
)

# Export to CSV
history.export_to_csv('history.csv')
```

### 5. PDF Report Generation

```python
from utils import ReportGenerator

detection_data = {
    'timestamp': '2024-02-18T10:30:00',
    'disease': 'Tomato___Late_blight',
    'confidence': 95.5,
    'features': {
        'Color': 'RGB(120, 150, 80)',
        'Texture': 'High edge density',
        'Size': '1920x1080'
    },
    'treatment': 'Apply mancozeb + copper fungicide...'
}

ReportGenerator.generate_pdf_report(detection_data, 'report.pdf')
```

## Configuration

### app_config.json
```json
{
  "location": "Kisii University",
  "max_history": 100,
  "confidence_threshold": 70,
  "enable_voice": true,
  "auto_enhance_images": true,
  "max_agrovet_results": 5,
  "batch_size": 32,
  "model_path": "crop_disease_cnn_model.keras"
}
```

### config.py
Modify constants in `config.py` for system-wide changes:
- Image size
- Batch size
- Directory paths
- UI settings
- Voice settings

## Migration from Old Version

### If you have existing detection_history.json
The new system is backward compatible. Your history will be automatically loaded.

### If you have custom models
1. Ensure they are in `.keras` format (not `.h5`)
2. Update model path in config
3. Test with `test_system.py`

### If you have custom class names
Update the class names list in your code:
```python
from main import PLANTVILLAGE_CLASSES, DEMO_CLASSES

# Use your custom classes
MY_CLASSES = ['Class1', 'Class2', ...]
```

## Performance Optimization

### Enable GPU
If you have an NVIDIA GPU:
```bash
pip install tensorflow-gpu
```

The system will automatically detect and use GPU.

### Adjust Batch Size
For faster processing with more RAM:
```python
# In config.py
BATCH_SIZE = 64  # Increase if you have more RAM
```

### Parallel Processing
For batch operations:
```python
processor = BatchProcessor(
    model_path='model.keras',
    class_names=classes,
    max_workers=8  # Increase for more CPU cores
)
```

## Troubleshooting

### Issue: Import errors
**Solution**: Run `pip install -r requirements.txt`

### Issue: Model not loading
**Solution**: 
1. Delete old `.h5` files
2. Run the app - it will create a demo model
3. Or train a new model with `model_trainer.py`

### Issue: Slow performance
**Solution**:
1. Enable GPU if available
2. Reduce image size in config
3. Use batch processing for multiple images
4. Close other applications

### Issue: Camera not working
**Solution**:
1. Check camera permissions
2. Try different camera index (0, 1, 2)
3. Update OpenCV: `pip install --upgrade opencv-python`

### Issue: Voice not working
**Solution**:
1. Install pyttsx3: `pip install pyttsx3`
2. Check audio drivers
3. Toggle voice in the app

## Best Practices

### For Best Accuracy
1. Use well-lit, clear images
2. Enable image enhancement
3. Ensure leaves are clearly visible
4. Use high-resolution images (but not too large)

### For Best Performance
1. Use GPU if available
2. Process images in batches
3. Cache frequently used models
4. Use appropriate image sizes

### For Production Use
1. Train custom models on your specific data
2. Set appropriate confidence thresholds
3. Enable logging for debugging
4. Regular model updates with new data

## Advanced Usage

### Custom Model Architecture
```python
from model_trainer import ModelTrainer

trainer = ModelTrainer(data_dir='data/train')

# Use transfer learning with MobileNetV2
trainer.build_model(num_classes=38, architecture='mobilenet')

# Or ResNet50 for better accuracy
trainer.build_model(num_classes=38, architecture='resnet')
```

### Ensemble Predictions
```python
# Load multiple models
model1 = models.load_model('model1.keras')
model2 = models.load_model('model2.keras')

# Get predictions from both
pred1 = model1.predict(image)
pred2 = model2.predict(image)

# Average predictions
ensemble_pred = (pred1 + pred2) / 2
```

### Custom Preprocessing Pipeline
```python
from utils import ImagePreprocessor

preprocessor = ImagePreprocessor()

def custom_preprocess(image):
    # Apply multiple enhancements
    image = preprocessor.enhance_image(image)
    image = preprocessor.remove_noise(image)
    image = preprocessor.auto_adjust_brightness(image)
    return image
```

## Support

For issues or questions:
1. Check this guide
2. Review README.md
3. Run test_system.py
4. Check logs in logs/ directory
5. Open an issue on GitHub

## Changelog

### Version 2.0 (Current)
- Added batch processing
- Added model training utilities
- Added PDF report generation
- Enhanced image preprocessing
- Improved agrovet finder
- Better error handling
- Comprehensive testing suite
- Full documentation

### Version 1.0 (Original)
- Basic GUI application
- Camera and upload support
- Disease detection
- Treatment recommendations
- Agrovet finder
- Voice assistant

## Next Steps

1. ✅ Install new dependencies
2. ✅ Run system tests
3. ✅ Try the enhanced application
4. ✅ Explore batch processing
5. ✅ Train custom models
6. ✅ Generate reports
7. ✅ Optimize for your use case

---

Happy farming! 🌾
