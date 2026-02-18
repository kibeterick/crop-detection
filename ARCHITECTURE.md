# System Architecture

## Overview
```
┌─────────────────────────────────────────────────────────────┐
│                  CROP DISEASE DETECTION SYSTEM              │
│                     Enhanced Architecture                    │
└─────────────────────────────────────────────────────────────┘
```

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Camera   │  │   Upload   │  │  Agrovet   │  │  Voice   │ │
│  │   Feed     │  │   Image    │  │   Finder   │  │ Assistant│ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MAIN APPLICATION                            │
│                         (main.py)                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CropHealthApp                                           │  │
│  │  - GUI Management                                        │  │
│  │  - Event Handling                                        │  │
│  │  - User Interaction                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CORE MODULES                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   config.py  │  │   utils.py   │  │batch_processor│         │
│  │              │  │              │  │     .py       │         │
│  │ Configuration│  │  Utilities   │  │    Batch      │         │
│  │  Management  │  │  & Helpers   │  │  Processing   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │model_trainer │  │test_system.py│  │              │         │
│  │    .py       │  │              │  │   (Future)   │         │
│  │   Model      │  │   Testing    │  │              │         │
│  │  Training    │  │    Suite     │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ImagePreprocessor                                       │  │
│  │  - CLAHE Enhancement                                     │  │
│  │  - Noise Reduction                                       │  │
│  │  - Brightness Adjustment                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ModelManager                                            │  │
│  │  - Model Loading                                         │  │
│  │  - Model Validation                                      │  │
│  │  - Prediction                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  HistoryManager                                          │  │
│  │  - Detection History                                     │  │
│  │  - Statistics                                            │  │
│  │  - Search & Filter                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ReportGenerator                                         │  │
│  │  - PDF Reports                                           │  │
│  │  - CSV Export                                            │  │
│  │  - Data Formatting                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  TensorFlow  │  │    OpenCV    │  │    Geopy     │         │
│  │   (Models)   │  │   (Vision)   │  │  (Location)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   pyttsx3    │  │  ReportLab   │  │    Pandas    │         │
│  │   (Voice)    │  │    (PDF)     │  │    (Data)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Single Image Detection
```
User Input (Camera/Upload)
    │
    ▼
Image Capture/Load
    │
    ▼
Image Preprocessing
    │
    ├─► CLAHE Enhancement
    ├─► Noise Reduction
    └─► Brightness Adjustment
    │
    ▼
Model Prediction
    │
    ├─► Feature Extraction
    ├─► CNN Processing
    └─► Confidence Calculation
    │
    ▼
Result Processing
    │
    ├─► Treatment Lookup
    ├─► History Recording
    └─► Voice Feedback
    │
    ▼
Display Results
    │
    ├─► GUI Update
    ├─► PDF Report (optional)
    └─► CSV Export (optional)
```

### Batch Processing
```
Folder Input
    │
    ▼
File Discovery
    │
    ▼
Parallel Processing
    │
    ├─► Worker 1 ─► Image 1 ─► Preprocess ─► Predict
    ├─► Worker 2 ─► Image 2 ─► Preprocess ─► Predict
    ├─► Worker 3 ─► Image 3 ─► Preprocess ─► Predict
    └─► Worker N ─► Image N ─► Preprocess ─► Predict
    │
    ▼
Result Aggregation
    │
    ├─► Statistics Calculation
    ├─► Summary Generation
    └─► Export (CSV/PDF)
    │
    ▼
Display Summary
```

### Model Training
```
Training Data
    │
    ▼
Data Augmentation
    │
    ├─► Rotation
    ├─► Shift
    ├─► Zoom
    └─► Flip
    │
    ▼
Model Architecture Selection
    │
    ├─► Simple CNN
    ├─► MobileNetV2
    └─► ResNet50
    │
    ▼
Training Loop
    │
    ├─► Forward Pass
    ├─► Loss Calculation
    ├─► Backpropagation
    └─► Weight Update
    │
    ▼
Validation
    │
    ├─► Accuracy Check
    ├─► Early Stopping
    └─► Best Model Save
    │
    ▼
Evaluation & Export
```

## Module Dependencies

```
main.py
├── config.py
├── utils.py
│   ├── ImagePreprocessor
│   ├── HistoryManager
│   └── ReportGenerator
├── batch_processor.py
│   ├── config.py
│   └── utils.py
└── model_trainer.py
    └── config.py

test_system.py
└── (tests all modules)
```

## File Organization

```
project_root/
│
├── Core Application
│   ├── main.py                 # Main GUI application
│   ├── config.py               # Configuration
│   └── utils.py                # Utilities
│
├── Advanced Features
│   ├── batch_processor.py      # Batch processing
│   └── model_trainer.py        # Model training
│
├── Testing & Installation
│   ├── test_system.py          # System tests
│   └── install.bat             # Installer
│
├── Documentation
│   ├── README.md               # Main documentation
│   ├── ENHANCEMENTS.md         # Enhancement details
│   ├── UPGRADE_GUIDE.md        # Upgrade instructions
│   ├── QUICK_REFERENCE.md      # Quick reference
│   ├── ARCHITECTURE.md         # This file
│   └── ENHANCEMENT_SUMMARY.txt # Summary
│
├── Configuration
│   ├── requirements.txt        # Dependencies
│   └── app_config.json         # User config
│
├── Data
│   ├── data/                   # Dataset
│   ├── logs/                   # Application logs
│   ├── debug_images/           # Debug output
│   └── detection_history.json  # History
│
└── Models
    ├── demo_model.keras        # Demo model
    └── *.keras                 # Custom models
```

## Class Hierarchy

```
CropHealthApp (main.py)
├── ModelManager
│   ├── validate_model_file()
│   ├── create_demo_model()
│   └── ensure_model_exists()
│
├── VoiceAssistant
│   ├── speak()
│   └── toggle_voice()
│
└── AgrovetFinder
    ├── get_location_coordinates()
    ├── find_nearest_agrovets()
    └── open_in_maps()

ImagePreprocessor (utils.py)
├── enhance_image()
├── remove_noise()
└── auto_adjust_brightness()

HistoryManager (utils.py)
├── load_history()
├── save_history()
├── add_record()
├── get_statistics()
├── search()
└── export_to_csv()

ReportGenerator (utils.py)
└── generate_pdf_report()

BatchProcessor (batch_processor.py)
├── load_model()
├── process_image()
├── process_folder()
├── get_summary()
└── export_results()

ModelTrainer (model_trainer.py)
├── configure_gpu()
├── create_data_generators()
├── build_model()
│   ├── _build_simple_cnn()
│   ├── _build_mobilenet()
│   └── _build_resnet()
├── train()
├── plot_training_history()
└── evaluate()
```

## Technology Stack

```
┌─────────────────────────────────────┐
│         Frontend Layer              │
│  ┌──────────────────────────────┐  │
│  │  Tkinter (GUI)               │  │
│  │  PIL/ImageTk (Image Display) │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│       Application Layer             │
│  ┌──────────────────────────────┐  │
│  │  Python 3.8+                 │  │
│  │  Threading (Async)           │  │
│  │  JSON (Config)               │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│      Processing Layer               │
│  ┌──────────────────────────────┐  │
│  │  TensorFlow/Keras (ML)       │  │
│  │  OpenCV (Vision)             │  │
│  │  NumPy (Computation)         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         Service Layer               │
│  ┌──────────────────────────────┐  │
│  │  Geopy (Geolocation)         │  │
│  │  pyttsx3 (Voice)             │  │
│  │  Requests (HTTP)             │  │
│  │  ReportLab (PDF)             │  │
│  │  Pandas (Data)               │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Execution Flow

### Application Startup
1. Load configuration (config.py)
2. Setup logging
3. Initialize GUI (Tkinter)
4. Load/create model
5. Initialize voice assistant
6. Initialize agrovet finder
7. Display main window

### Image Analysis
1. Capture/load image
2. Preprocess image
3. Extract features
4. Run CNN prediction
5. Get top predictions
6. Lookup treatment
7. Update GUI
8. Save to history
9. Voice feedback

### Batch Processing
1. Scan folder for images
2. Create worker pool
3. Process images in parallel
4. Collect results
5. Generate statistics
6. Export to CSV/PDF

### Model Training
1. Load training data
2. Create data generators
3. Build model architecture
4. Configure callbacks
5. Train model
6. Validate performance
7. Save best model
8. Plot training history

## Security Considerations

- Input validation for all user inputs
- File type validation for uploads
- Path sanitization
- Error handling for external services
- Secure model loading
- Safe file operations

## Performance Optimization

- GPU memory growth configuration
- Parallel batch processing
- Image caching
- Model prediction caching
- Efficient data structures
- Lazy loading where possible

## Extensibility Points

1. **New Model Architectures**
   - Add to ModelTrainer._build_*() methods

2. **New Preprocessing Techniques**
   - Add to ImagePreprocessor class

3. **New Export Formats**
   - Add to ReportGenerator class

4. **New Data Sources**
   - Extend BatchProcessor

5. **New UI Components**
   - Add to CropHealthApp

---

This architecture provides a solid foundation for future enhancements
while maintaining clean separation of concerns and modularity.
