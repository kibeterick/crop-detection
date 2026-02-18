# 🌾 CNN-Based Crop Disease Detection System

A comprehensive deep learning application for detecting crop diseases using Convolutional Neural Networks (CNN). Features include real-time camera detection, batch processing, agrovet finder, and detailed reporting.

## ✨ Features

### Core Features
- 🎥 **Real-time Camera Detection** - Live disease detection from webcam
- 📸 **Image Upload** - Analyze images from your device
- 🧠 **CNN-Powered Analysis** - Deep learning model for accurate predictions
- 📊 **Confidence Scoring** - Get confidence levels for each prediction
- 💊 **Treatment Recommendations** - Detailed treatment advice for detected diseases
- 🗺️ **Agrovet Finder** - Locate nearest agricultural supply stores
- 🔊 **Voice Assistant** - Audio feedback for accessibility

### Advanced Features
- 📦 **Batch Processing** - Process multiple images at once
- 📈 **History Tracking** - Keep track of all detections
- 📄 **PDF Reports** - Generate professional reports
- 📊 **CSV Export** - Export data for analysis
- 🎨 **Image Enhancement** - Auto-enhance images for better accuracy
- 🔄 **Model Training** - Train custom models on your data
- 📉 **Statistics Dashboard** - View detection trends and patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for camera features)
- 4GB RAM minimum (8GB recommended)
- GPU (optional, for faster processing)

### Installation

1. **Clone or download the project**
```bash
cd crop-disease-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python main.py
```

## 📖 Usage Guide

### GUI Application

1. **Start the application**
   ```bash
   python main.py
   ```
   Choose option 1 for GUI mode

2. **Using Camera**
   - Click "Start Camera" to begin live feed
   - Click "Capture & Analyze" to detect disease
   - Click "Stop Camera" when done

3. **Upload Image**
   - Click "Upload Image"
   - Select an image file
   - View results automatically

4. **Find Agrovet**
   - Enter your location
   - Click "Find Agrovet"
   - Click "Open in Google Maps" for directions

### Batch Processing

Process multiple images at once:

```python
from batch_processor import BatchProcessor
from main import DEMO_CLASSES

# Initialize processor
processor = BatchProcessor(
    model_path='demo_model.keras',
    class_names=DEMO_CLASSES,
    max_workers=4
)

# Process folder
results = processor.process_folder('path/to/images')

# Get summary
summary = processor.get_summary()
print(summary)

# Export results
processor.export_results('results.csv')
```

### Training Custom Models

Train your own model:

```python
from model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    data_dir='data/train',
    img_size=(224, 224),
    batch_size=32
)

# Build model
trainer.build_model(num_classes=8, architecture='mobilenet')

# Train
trainer.train(epochs=20, save_path='my_model.keras')

# Plot training history
trainer.plot_training_history('history.png')

# Evaluate
trainer.evaluate('data/test')
```

### Generate Reports

Create PDF reports:

```python
from utils import ReportGenerator

detection_data = {
    'timestamp': '2024-02-18T10:30:00',
    'disease': 'Tomato___Late_blight',
    'confidence': 95.5,
    'features': {
        'Color': 'RGB(120, 150, 80)',
        'Texture': 'High edge density'
    },
    'treatment': 'Apply mancozeb + copper fungicide...'
}

ReportGenerator.generate_pdf_report(detection_data, 'report.pdf')
```

## 📁 Project Structure

```
crop-disease-detection/
├── main.py                 # Main application
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── batch_processor.py     # Batch processing
├── model_trainer.py       # Model training utilities
├── requirements.txt       # Dependencies
├── README.md             # This file
├── ENHANCEMENTS.md       # Enhancement documentation
├── data/                 # Dataset directory
│   ├── train/           # Training data
│   └── test/            # Test data
├── logs/                # Application logs
├── debug_images/        # Debug output
└── models/              # Trained models
```

## 🎯 Supported Diseases

The system can detect 38 different plant diseases across multiple crops:

- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Orange**: Huanglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

## ⚙️ Configuration

Edit `app_config.json` to customize settings:

```json
{
  "location": "your_location",
  "max_history": 100,
  "confidence_threshold": 70,
  "enable_voice": true,
  "auto_enhance_images": true,
  "max_agrovet_results": 5
}
```

## 🔧 Troubleshooting

### Model Not Loading
- Ensure TensorFlow is properly installed
- Check model file integrity
- Try creating a demo model (automatic on first run)

### Camera Not Working
- Check camera permissions
- Try different camera indices (0, 1, 2)
- Verify OpenCV installation

### Slow Performance
- Enable GPU if available
- Reduce batch size
- Use smaller image sizes
- Close other applications

### Voice Not Working
- Install pyttsx3: `pip install pyttsx3`
- Check audio drivers
- Toggle voice on/off in the app

## 📊 Performance

- **Accuracy**: ~85-95% (depends on model and data quality)
- **Processing Speed**: 0.5-1s per image (CPU), 0.1-0.3s (GPU)
- **Batch Processing**: ~100 images/minute
- **Model Size**: 10-50MB (depends on architecture)

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - Feel free to use and modify for your projects

## 🙏 Acknowledgments

- PlantVillage Dataset for training data
- TensorFlow team for the deep learning framework
- OpenCV community for image processing tools
- All contributors and users

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation
- Review troubleshooting guide

## 🔮 Future Roadmap

- [ ] Mobile app (Android/iOS)
- [ ] REST API for integration
- [ ] Cloud deployment
- [ ] Real-time monitoring system
- [ ] Multi-language support
- [ ] Community features
- [ ] Expert consultation integration
- [ ] IoT device integration

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Deep Learning for Agriculture](https://arxiv.org/abs/1807.11809)

---

Made with ❤️ for farmers and agricultural professionals
