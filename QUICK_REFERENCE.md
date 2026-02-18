# Quick Reference Guide

## Installation
```bash
pip install -r requirements.txt
python main.py
```

## Common Commands

### Run Application
```bash
python main.py          # GUI mode
python test_system.py   # Test system
```

### Batch Processing
```python
from batch_processor import BatchProcessor
processor = BatchProcessor('model.keras', class_names)
results = processor.process_folder('images/')
processor.export_results('results.csv')
```

### Train Model
```python
from model_trainer import ModelTrainer
trainer = ModelTrainer('data/train')
trainer.build_model(num_classes=38, architecture='mobilenet')
trainer.train(epochs=20, save_path='model.keras')
```

### Generate Report
```python
from utils import ReportGenerator
ReportGenerator.generate_pdf_report(data, 'report.pdf')
```

### History Management
```python
from utils import HistoryManager
history = HistoryManager()
stats = history.get_statistics()
history.export_to_csv('history.csv')
```

## File Structure
```
main.py              - Main application
config.py            - Configuration
utils.py             - Utilities
batch_processor.py   - Batch processing
model_trainer.py     - Model training
test_system.py       - System tests
requirements.txt     - Dependencies
```

## Key Features
- 🎥 Real-time camera detection
- 📸 Image upload analysis
- 📦 Batch processing
- 🎓 Custom model training
- 📄 PDF report generation
- 🗺️ Agrovet finder
- 🔊 Voice assistant
- 📊 History tracking

## Keyboard Shortcuts (GUI)
- Ctrl+O: Upload image
- Ctrl+S: Start camera
- Ctrl+Q: Quit
- Ctrl+V: Toggle voice

## Configuration (app_config.json)
```json
{
  "location": "your_location",
  "max_history": 100,
  "confidence_threshold": 70,
  "enable_voice": true
}
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| Model not loading | Delete old .h5 files, restart app |
| Camera not working | Check permissions, try index 0-2 |
| Slow performance | Enable GPU, reduce batch size |
| Voice not working | `pip install pyttsx3` |

## Performance Tips
- Use GPU for faster processing
- Enable image enhancement
- Process images in batches
- Use appropriate image sizes (224x224)
- Close unnecessary applications

## Supported Crops
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## Model Architectures
- **simple**: Fast, lightweight (10-20MB)
- **mobilenet**: Balanced (20-30MB)
- **resnet**: High accuracy (50-100MB)

## Export Formats
- CSV: Tabular data
- PDF: Professional reports
- JSON: Detection history

## API Quick Reference

### ImagePreprocessor
```python
preprocessor.enhance_image(img)
preprocessor.remove_noise(img)
preprocessor.auto_adjust_brightness(img)
```

### HistoryManager
```python
history.add_record(disease, confidence, features, treatment)
history.get_statistics()
history.search(disease='Tomato', min_confidence=80)
history.export_to_csv('file.csv')
```

### BatchProcessor
```python
processor.process_image('image.jpg')
processor.process_folder('folder/')
processor.get_summary()
processor.export_results('results.csv')
```

### ModelTrainer
```python
trainer.build_model(num_classes, architecture)
trainer.train(epochs, save_path)
trainer.plot_training_history('plot.png')
trainer.evaluate('test_dir')
```

## Common Workflows

### Workflow 1: Single Image Analysis
1. Run `python main.py`
2. Click "Upload Image"
3. Select image
4. View results
5. Find nearest agrovet

### Workflow 2: Batch Processing
1. Organize images in folder
2. Run batch processor
3. Review results
4. Export to CSV

### Workflow 3: Custom Model Training
1. Prepare training data
2. Run model trainer
3. Evaluate performance
4. Use in main app

### Workflow 4: Generate Report
1. Detect disease
2. Collect data
3. Generate PDF report
4. Share with experts

## Environment Variables
```bash
TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
CUDA_VISIBLE_DEVICES=0   # Select GPU
```

## Dependencies
- tensorflow >= 2.13.0
- opencv-python >= 4.8.0
- pillow >= 10.0.0
- numpy >= 1.24.0
- geopy >= 2.4.0
- pyttsx3 >= 2.90
- requests >= 2.31.0
- reportlab >= 4.0.0 (optional)
- pandas >= 2.0.0 (optional)

## Resources
- README.md - Full documentation
- ENHANCEMENTS.md - Enhancement details
- UPGRADE_GUIDE.md - Upgrade instructions
- test_system.py - System diagnostics

## Support
1. Run `python test_system.py`
2. Check logs in `logs/` directory
3. Review documentation
4. Open GitHub issue

---
For detailed information, see README.md
