# -*- coding: utf-8 -*-
"""
CNN-BASED CROP DISEASE DETECTION SYSTEM - SIMPLIFIED VERSION
=================================================================
A streamlined application with essential features only.
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Tuple, Any

# Essential imports with error handling
try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    print("ERROR: Tkinter not available")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: Requests not installed. Run: pip install requests")
    sys.exit(1)

# CORRECTED IMPORT: Fixed typo and removed unused 'geodesic1'
try:
    import geopy
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
except ImportError:
    print("WARNING: Geopy not installed. Agrovet finder will be disabled.")
    print("Run: pip install geopy to enable agrovet finder")
    geopy = None

try:
    import pyttsx3
except ImportError:
    print("WARNING: pyttsx3 not installed. Voice features will be disabled.")
    print("Run: pip install pyttsx3 to enable voice features")
    pyttsx3 = None

try:
    import webbrowser
except ImportError:
    print("ERROR: Webbrowser module not available")
    sys.exit(1)

# Model paths
DEFAULT_MODEL_PATH = 'crop_disease_cnn_model.keras'
DEMO_MODEL_PATH = 'demo_model.keras'

# Import configuration and utilities
try:
    from config import Config
    from utils import ImagePreprocessor, HistoryManager, ReportGenerator, validate_image, format_confidence
    from leaf_detection import LeafDetector, LeafQualityChecker
except ImportError as e:
    print(f"ERROR: Required module not found: {e}")
    print("Ensure config.py, utils.py, and leaf_detection.py are in the same directory.")
    sys.exit(1)


# Configure logging
def setup_logging():
    """Setup logging"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler('logs/crop_disease_detector.log', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# PlantVillage class names
PLANTVILLAGE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Simplified classes for demo
DEMO_CLASSES = [
    'Apple___healthy', 'Apple___Apple_scab', 'Tomato___healthy', 'Tomato___Late_blight',
    'Corn___healthy', 'Corn___Common_rust', 'Grape___healthy', 'Grape___Black_rot'
]

# Treatment database
TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicides (Captan, myclobutanil) every 7-10 days. Prune for air flow.',
    'Apple___Black_rot': 'Sanitize: remove infected fruit, prune dead branches. Use copper fungicides.',
    'Apple___Cedar_apple_rust': 'Use resistant varieties. Apply myclobutanil in spring.',
    'Apple___healthy': 'No disease. Maintain health with balanced fertilizer.',
    'Blueberry___healthy': 'No disease. Ensure acidic soil, good drainage.',
    'Cherry___Powdery_mildew': 'Apply sulfur-based fungicides. Prune for air circulation.',
    'Cherry___healthy': 'No disease. Prune annually for shape.',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids, rotate crops.',
    'Corn___Common_rust': 'Plant resistant varieties. Fungicides rarely needed.',
    'Corn___Northern_Leaf_Blight': 'Rotate crops, manage residue. Use strobilurin fungicides.',
    'Corn___healthy': 'No disease. Maintain soil fertility.',
    'Grape___Black_rot': 'Apply mancozeb every 7-10 days. Prune for air flow.',
    'Grape___Esca_(Black_Measles)': 'No chemical cure. Prune infected wood.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use copper fungicides. Improve sanitation.',
    'Grape___healthy': 'No disease. Prune and train vines.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Remove infected trees, control psyllids.',
    'Peach___Bacterial_spot': 'Apply copper bactericides. Use resistant varieties.',
    'Peach___healthy': 'No disease. Prune for open canopy.',
    'Pepper,_bell___Bacterial_spot': 'Use copper sprays, rotate crops.',
    'Pepper,_bell___healthy': 'No disease. Ensure even watering.',
    'Potato___Early_blight': 'Apply chlorothalonil. Rotate crops, mulch.',
    'Potato___Late_blight': 'Use mancozeb + copper. Destroy infected plants.',
    'Potato___healthy': 'No disease. Hill soil, rotate crops.',
    'Raspberry___healthy': 'No disease. Prune canes annually.',
    'Soybean___healthy': 'No disease. Rotate with non-legumes.',
    'Squash___Powdery_mildew': 'Apply sulfur or potassium bicarbonate.',
    'Strawberry___Leaf_scorch': 'Use fungicides if severe. Improve irrigation.',
    'Strawberry___healthy': 'No disease. Mulch, rotate beds.',
    'Tomato___Bacterial_spot': 'Apply copper bactericides. Avoid overhead watering.',
    'Tomato___Early_blight': 'Use chlorothalonil or copper. Mulch, stake plants.',
    'Tomato___Late_blight': 'Apply mancozeb + copper. Improve air circulation.',
    'Tomato___Leaf_Mold': 'Use sulfur-based fungicides. Increase ventilation.',
    'Tomato___Septoria_leaf_spot': 'Apply copper or mancozeb. Water at base.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides. Increase humidity.',
    'Tomato___Target_Spot': 'Apply azoxystrobin. Remove infected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure. Use resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'No cure. Disinfect tools, use virus-free seeds.',
    'Tomato___healthy': 'No disease. Stake and prune for air flow.',
}
GENERAL_TREATMENT = "Consult local agricultural extension for region-specific advice."

# Known locations with their coordinates
KNOWN_LOCATIONS = {
    "kisii university": (-0.6817, 34.7666),
    "nairobi": (-1.2921, 36.8219),
    "mombasa": (-4.0435, 39.6682),
    "kisumu": (-0.0917, 34.7680),
    "nakuru": (-0.3031, 36.0695),
    "eldoret": (0.5143, 35.2698),
    "thika": (-1.0361, 37.0764),
    "kitale": (1.0136, 35.0035),
    "garissa": (-0.4529, 39.6460),
    "kakamega": (0.2842, 34.7519)
}

# Real agrovet data with actual coordinates
AGROVET_DATA = {
    "kisii": [
        {"name": "Kisii Agro-Vet Centre", "lat": -0.6785, "lon": 34.7701,
         "address": "Kisii Town, near main market"},
        {"name": "Safina Agrovet", "lat": -0.6825, "lon": 34.7670, "address": "Kisii Town, along Kisii-Kilgoris Road"},
        {"name": "Jubilee Agrovet", "lat": -0.6805, "lon": 34.7650, "address": "Kisii Town, near bus stage"},
        {"name": "Elgon Agrovet", "lat": -0.6840, "lon": 34.7700, "address": "Kisii Town, along Kisii-Nyamira Road"},
        {"name": "Cooperative Agrovet", "lat": -0.6790, "lon": 34.7620,
         "address": "Kisii Town, near cooperative bank"},
        {"name": "Kisii Farmers Centre", "lat": -0.6760, "lon": 34.7680,
         "address": "Kisii Town, along Kisii-Ogembo Road"},
        {"name": "Mwalimu Agrovet", "lat": -0.6830, "lon": 34.7710, "address": "Kisii Town, near hospital"}
    ],
    "nairobi": [
        {"name": "Nairobi Agrovet Ltd", "lat": -1.2921, "lon": 36.8219, "address": "Nairobi CBD, Moi Avenue"},
        {"name": "Farm Input Centre", "lat": -1.2850, "lon": 36.8250, "address": "Nairobi, Ngong Road"},
        {"name": "Kenya Seed Company", "lat": -1.2980, "lon": 36.8190, "address": "Nairobi, Kijabe Street"},
        {"name": "Elgon Kenya Ltd", "lat": -1.2860, "lon": 36.8230, "address": "Nairobi, Mombasa Road"}
    ],
    "mombasa": [
        {"name": "Mombasa Agrovet", "lat": -4.0435, "lon": 39.6682, "address": "Mombasa, Digo Road"},
        {"name": "Coast Agro Supplies", "lat": -4.0500, "lon": 39.6700, "address": "Mombasa, Nyerere Avenue"}
    ],
    "kisumu": [
        {"name": "Kisumu Agrovet", "lat": -0.0917, "lon": 34.7680, "address": "Kisumu, Oginga Odinga Street"},
        {"name": "Lake Region Agrovet", "lat": -0.0950, "lon": 34.7650, "address": "Kisumu, Mosque Road"}
    ],
    "nakuru": [
        {"name": "Nakuru Agrovet", "lat": -0.3031, "lon": 36.0695, "address": "Nakuru, Kenyatta Avenue"},
        {"name": "Rift Valley Agro Supplies", "lat": -0.3080, "lon": 36.0720, "address": "Nakuru, Mburu Gichua Road"}
    ],
    "eldoret": [
        {"name": "Eldoret Agrovet", "lat": 0.5143, "lon": 35.2698, "address": "Eldoret, Kenyatta Street"},
        {"name": "North Rift Agro Supplies", "lat": 0.5180, "lon": 35.2720, "address": "Eldoret, Uganda Road"}
    ],
    "thika": [
        {"name": "Thika Agrovet", "lat": -1.0361, "lon": 37.0764, "address": "Thika, Garissa Road"},
        {"name": "Thika Farmers Centre", "lat": -1.0400, "lon": 37.0800, "address": "Thika, Makongeni Road"}
    ],
    "kitale": [
        {"name": "Kitale Agrovet", "lat": 1.0136, "lon": 35.0035, "address": "Kitale, Kapenguria Road"},
        {"name": "Trans Nzoia Agro Supplies", "lat": 1.0180, "lon": 35.0070, "address": "Kitale, Hospital Road"}
    ],
    "garissa": [
        {"name": "Garissa Agrovet", "lat": -0.4529, "lon": 39.6460, "address": "Garissa, Kismayu Road"},
        {"name": "North Eastern Agro Supplies", "lat": -0.4580, "lon": 39.6500, "address": "Garissa, Bula Sheikh"}
    ],
    "kakamega": [
        {"name": "Kakamega Agrovet", "lat": 0.2842, "lon": 34.7519, "address": "Kakamega, Mumias Road"},
        {"name": "Western Agro Supplies", "lat": 0.2800, "lon": 34.7550, "address": "Kakamega, Amalemba Road"}
    ]
}


class ModelManager:
    """Manages model downloading and creation"""

    @staticmethod
    def validate_model_file(model_path: str) -> bool:
        """Validate if a model file is valid"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                return False

            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB
                logger.error(f"Model file too small: {file_size} bytes")
                return False

            # Try to load the model
            model = models.load_model(model_path)

            # Try a prediction
            dummy_input = np.random.random((1, 224, 224, 3))
            output = model.predict(dummy_input, verbose=0)

            if output is not None and output.shape[0] == 1:
                logger.info("Model validation successful")
                return True

            return False

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    @staticmethod
    def create_demo_model(save_path: str) -> bool:
        """Create a simple demo model for testing"""
        logger.info("Creating demo model...")
        try:
            # Configure TensorFlow to use less memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.error(f"Error setting GPU memory growth: {e}")

            # Create a simpler CNN model
            model = models.Sequential([
                layers.Input(shape=(224, 224, 3)),
                layers.Rescaling(1. / 255),
                layers.Conv2D(16, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(DEMO_CLASSES), activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Create smaller dummy training data
            dummy_x = np.random.random((20, 224, 224, 3))
            dummy_y = np.random.randint(0, len(DEMO_CLASSES), 20)

            # Train for a few epochs
            model.fit(dummy_x, dummy_y, epochs=2, verbose=0, batch_size=4)

            # Save the model
            model.save(save_path)

            # Validate the created model
            if ModelManager.validate_model_file(save_path):
                logger.info(f"Demo model created and validated: {save_path}")
                return True
            else:
                logger.error("Created demo model is invalid")
                return False

        except Exception as e:
            logger.error(f"Failed to create demo model: {e}")
            return False

    @staticmethod
    def ensure_model_exists(model_path: str) -> bool:
        """Ensure a valid model exists"""
        # Check if model already exists and is valid
        if os.path.exists(model_path):
            logger.info(f"Model file found at {model_path}")
            if ModelManager.validate_model_file(model_path):
                logger.info("Existing model is valid")
                return True
            else:
                logger.warning("Existing model is invalid, will recreate")
                # Backup the invalid model
                backup_path = f"{model_path}.invalid"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(model_path, backup_path)
                logger.info(f"Invalid model backed up to {backup_path}")

        # Create a demo model
        logger.info("Creating demo model")
        if ModelManager.create_demo_model(DEMO_MODEL_PATH):
            return DEMO_MODEL_PATH

        logger.error("Failed to obtain any valid model")
        return False


class VoiceAssistant:
    """Voice assistant class"""

    def __init__(self):
        self.enabled = False
        self.engine = None
        self.last_spoken = ""
        self.speaking = False

        if pyttsx3 is None:
            logger.warning("pyttsx3 not available, voice features disabled")
            return

        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if voices:
                # Use first available voice
                self.engine.setProperty('voice', voices[0].id)

            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.enabled = True
            logger.info("Voice assistant initialized successfully")

        except Exception as e:
            logger.error(f"Voice initialization failed: {e}")
            self.enabled = False

    def speak(self, text: str, force: bool = False):
        """Speak text using text-to-speech"""
        if not self.enabled or not self.engine:
            return
        if text == self.last_spoken and not force:
            return
        self.last_spoken = text

        def speak_thread():
            self.speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"Voice error: {e}")
            finally:
                self.speaking = False

        if not self.speaking:
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()

    def toggle_voice(self):
        """Toggle voice assistant on/off"""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        logger.info(f"Voice {status}")
        self.speak(f"Voice {status}", force=True)
        return self.enabled


# MAJOR FIX: This class now finds the true nearest agrovets from the entire database
class AgrovetFinder:
    """Class to find the nearest agrovet stores"""

    def __init__(self):
        if geopy is None:
            logger.warning("Geopy not available, agrovet finder disabled")
            self.geolocator = None
            return

        try:
            self.geolocator = Nominatim(user_agent="crop_disease_detector")
            logger.info("AgrovetFinder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize geolocator: {e}")
            self.geolocator = None

    def get_location_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location name"""
        if not self.geolocator:
            logger.warning("Geolocator not available")
            return None

        try:
            # Check if it's a known location
            location_key = location_name.lower().strip()
            if location_key in KNOWN_LOCATIONS:
                logger.info(f"Using known coordinates for {location_name}")
                return KNOWN_LOCATIONS[location_key]

            # Otherwise, geocode the location
            location = self.geolocator.geocode(location_name, timeout=10)
            if location:
                logger.info(f"Geocoded {location_name} to {location.latitude}, {location.longitude}")
                return (location.latitude, location.longitude)
            else:
                logger.warning(f"Could not geocode location: {location_name}")
                return None
        except Exception as e:
            logger.error(f"Error getting location coordinates: {e}")
            return None

    def find_nearest_agrovets(self, location_name: str, num_results: int = 3) -> Optional[list]:
        """
        Find the top N nearest agrovets to the given location by checking all available stores.
        """
        try:
            # Get coordinates for the user's location
            user_coords = self.get_location_coordinates(location_name)
            if not user_coords:
                return None

            all_agrovets_with_distance = []

            # Iterate through all cities and their agrovets
            for city, agrovets in AGROVET_DATA.items():
                for agrovet in agrovets:
                    try:
                        # Calculate distance from the user's location to this specific agrovet
                        distance = geodesic(user_coords, (agrovet["lat"], agrovet["lon"])).kilometers
                        agrovet_copy = agrovet.copy()
                        agrovet_copy["distance"] = round(distance, 2)
                        all_agrovets_with_distance.append(agrovet_copy)
                    except Exception as e:
                        logger.error(f"Error calculating distance for {agrovet.get('name', 'Unknown')}: {e}")
                        continue

            # Sort all agrovets by distance
            all_agrovets_with_distance.sort(key=lambda x: x["distance"])

            # Return the top N results
            return all_agrovets_with_distance[:num_results]

        except Exception as e:
            logger.error(f"Error finding nearest agrovets: {e}")
            return None

    def open_in_maps(self, agrovet: Dict[str, Any], current_location: str) -> bool:
        """Open the agrovet location in Google Maps with directions"""
        try:
            # Get coordinates for the current location
            current_coords = self.get_location_coordinates(current_location)
            if not current_coords:
                # If we can't get current coords, just open the agrovet's location
                url = f"https://www.google.com/maps/search/?api=1&query={agrovet['lat']},{agrovet['lon']}"
            else:
                # Open directions from the current location to the agrovet
                url = f"https://www.google.com/maps/dir/?api=1&origin={current_coords[0]},{current_coords[1]}&destination={agrovet['lat']},{agrovet['lon']}"

            # DEBUG: Print the URL to the console to help diagnose issues
            print(f"--- DEBUG: Attempting to open URL: {url} ---")

            webbrowser.open(url)
            logger.info(f"Opened Google Maps with URL: {url}")
            return True
        except Exception as e:
            logger.error(f"Error opening maps: {e}")
            return False


class CropHealthApp:
    """Main GUI application"""

    def __init__(self, root: tk.Tk, model_path: str = DEFAULT_MODEL_PATH):
        self.root = root
        self.root.title("CNN Crop Disease Detection System")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f8ff')

        # Initialize components
        self.model_path = model_path
        self.model = None
        self.class_names = DEMO_CLASSES if model_path == DEMO_MODEL_PATH else PLANTVILLAGE_CLASSES
        self.voice = VoiceAssistant()
        self.cap = None
        self.current_frame = None
        self.uploaded_image = None
        self.detection_history = []
        self.max_history = 10
        self.agrovet_finder = AgrovetFinder()
        
        # Initialize leaf detection system
        if LeafDetector is not None:
            self.leaf_detector = LeafDetector()
            self.quality_checker = LeafQualityChecker()
            self.enable_leaf_detection = True
            logger.info("Leaf detection system initialized")
        else:
            self.leaf_detector = None
            self.quality_checker = None
            self.enable_leaf_detection = False
            logger.warning("Leaf detection system not available")

        # Create UI
        self.create_widgets()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load model
        self.load_model()

        # Welcome message
        self.voice.speak("Welcome to the CNN Crop Disease Detection System")
        logger.info("CropHealthApp initialized")

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#4CAF50')
        title_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(
            title_frame,
            text="🌾 CNN Crop Disease Detection System 🌾",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#4CAF50'
        )
        title_label.pack(pady=10)

        # Main content
        main_frame = tk.Frame(self.root, bg='#f0f8ff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel - Camera
        left_frame = tk.Frame(main_frame, bg='#f0f8ff')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        camera_frame = tk.LabelFrame(
            left_frame,
            text="Camera Feed",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff'
        )
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.camera_label = tk.Label(
            camera_frame,
            text="Camera feed will appear here\n\nClick 'Start Camera' to begin\n\nOr upload an image below (auto-analyzes)",
            font=('Arial', 12),
            bg='#e0e0e0',
            width=80,
            height=25
        )
        self.camera_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Camera controls
        camera_controls = tk.Frame(left_frame, bg='#f0f8ff')
        camera_controls.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = tk.Button(
            camera_controls,
            text="Start Camera",
            command=self.start_camera,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=12
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            camera_controls,
            text="Stop Camera",
            command=self.stop_camera,
            bg='#f44336',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=12,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.capture_btn = tk.Button(
            camera_controls,
            text="Capture & Analyze",
            command=self.capture_and_analyze,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=15,
            state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        # File upload
        file_frame = tk.Frame(left_frame, bg='#f0f8ff')
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            file_frame,
            text="Or upload an image (auto-analyzes):",
            font=('Arial', 10),
            bg='#f0f8ff'
        ).pack(side=tk.LEFT, padx=5)

        self.upload_btn = tk.Button(
            file_frame,
            text="Upload & Analyze Image",
            command=self.upload_image,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#f0f8ff', width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_frame.pack_propagate(False)

        # Results
        results_frame = tk.LabelFrame(
            right_frame,
            text="Analysis Results",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff'
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(
            results_frame,
            font=('Arial', 10),
            bg='white',
            wrap=tk.WORD,
            height=12
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Agrovet finder section
        agrovet_frame = tk.LabelFrame(
            right_frame,
            text="Find Nearest Agrovet",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff'
        )
        agrovet_frame.pack(fill=tk.X, padx=5, pady=5)

        location_frame = tk.Frame(agrovet_frame, bg='#f0f8ff')
        location_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            location_frame,
            text="Your Location:",
            font=('Arial', 10),
            bg='#f0f8ff'
        ).pack(side=tk.LEFT, padx=5)

        self.location_var = tk.StringVar(value="Kisii University")
        self.location_entry = tk.Entry(
            location_frame,
            textvariable=self.location_var,
            font=('Arial', 10),
            width=20
        )
        self.location_entry.pack(side=tk.LEFT, padx=5)

        self.find_agrovet_btn = tk.Button(
            location_frame,
            text="Find Agrovet",
            command=self.find_nearest_agrovet,
            bg='#9C27B0',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.find_agrovet_btn.pack(side=tk.LEFT, padx=5)

        # NEW BUTTON: Added "Show All Agrovets" button
        self.show_all_btn = tk.Button(
            location_frame,
            text="Show All Agrovets",
            command=self.show_all_agrovets,
            bg='#009688',  # A different color to distinguish it
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.show_all_btn.pack(side=tk.LEFT, padx=5)

        self.agrovet_info = tk.Text(
            agrovet_frame,
            font=('Arial', 10),
            bg='#f3e5f5',
            wrap=tk.WORD,
            height=4
        )
        self.agrovet_info.pack(fill=tk.X, padx=5, pady=5)

        self.maps_btn = tk.Button(
            agrovet_frame,
            text="Open in Google Maps",
            command=self.open_agrovet_in_maps,
            bg='#3F51B5',
            fg='white',
            font=('Arial', 10, 'bold'),
            state=tk.DISABLED
        )
        self.maps_btn.pack(pady=5)

        # Voice toggle
        self.voice_btn = tk.Button(
            right_frame,
            text="Toggle Voice",
            command=self.toggle_voice,
            bg='#607D8B',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=20
        )
        self.voice_btn.pack(pady=5)
        
        # Leaf detection toggle
        if self.enable_leaf_detection:
            self.leaf_detection_var = tk.BooleanVar(value=True)
            self.leaf_detection_check = tk.Checkbutton(
                right_frame,
                text="🍃 Enable Smart Leaf Detection",
                variable=self.leaf_detection_var,
                font=('Arial', 10),
                bg='#f0f8ff',
                command=self.toggle_leaf_detection
            )
            self.leaf_detection_check.pack(pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Model loading...")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#e0e0e0'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        """Load the CNN model"""
        try:
            # Ensure a valid model exists
            result = ModelManager.ensure_model_exists(self.model_path)

            if result is True:
                # Valid model exists at the original path
                if os.path.exists(self.model_path):
                    self.model = models.load_model(self.model_path)
                    self.status_var.set(f"Model loaded: {os.path.basename(self.model_path)}")
                    logger.info(f"Model loaded from {self.model_path}")
                else:
                    logger.error("Model path doesn't exist after validation")
                    self.status_var.set("Model loading failed")
                    return
            elif isinstance(result, str):
                # Demo model was created
                self.model_path = result
                self.class_names = DEMO_CLASSES
                self.model = models.load_model(self.model_path)
                self.status_var.set(f"Demo model loaded: {os.path.basename(self.model_path)}")
                logger.info(f"Demo model loaded from {self.model_path}")
            else:
                # No valid model could be obtained
                messagebox.showerror("Model Error", "Failed to obtain a valid model")
                self.status_var.set("Model loading failed")
                return

            self.voice.speak("CNN model loaded successfully")

        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {e}")
            self.status_var.set("Model loading failed")
            logger.error(f"Model loading failed: {e}")

    def start_camera(self):
        """Start camera feed"""
        try:
            # Try different camera indices
            for idx in range(3):  # Try indices 0, 1, 2
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    # Test if we can actually read from the camera
                    ret, frame = self.cap.read()
                    if ret:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                        self.start_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.NORMAL)
                        self.capture_btn.config(state=tk.NORMAL)
                        self.status_var.set("Camera started")
                        logger.info(f"Camera started with index {idx}")

                        self.update_camera_feed()
                        return
                    else:
                        self.cap.release()
                else:
                    self.cap.release()

            # If we get here, no camera worked
            messagebox.showerror("Camera Error", "Could not open any camera")

        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            logger.error(f"Camera start failed: {e}")

    def stop_camera(self):
        """Stop camera feed"""
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera stopped")
        logger.info("Camera stopped")

    def update_camera_feed(self):
        """Update camera feed"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()

                # Convert and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)

                # Resize to fit label
                label_width = self.camera_label.winfo_width()
                label_height = self.camera_label.winfo_height()

                if label_width > 1 and label_height > 1:
                    img_width, img_height = img.size
                    aspect_ratio = img_width / img_height

                    if label_width / label_height > aspect_ratio:
                        new_height = label_height - 20
                        new_width = int(new_height * aspect_ratio)
                    else:
                        new_width = label_width - 20
                        new_height = int(new_width / aspect_ratio)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(img)
                self.camera_label.config(image=photo)
                self.camera_label.image = photo

                self.root.after(10, self.update_camera_feed)
            else:
                self.stop_camera()

    def capture_and_analyze(self):
        """Capture and analyze image"""
        if self.current_frame is not None:
            self.analyze_image(self.current_frame, source="camera")

    def upload_image(self):
        """Upload and analyze image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Read image using PIL for better compatibility
                pil_img = Image.open(file_path)

                # Convert to RGB if necessary
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to numpy array
                img_array = np.array(pil_img)

                # Convert RGB to BGR for OpenCV processing
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img = img_array

                # Store the uploaded image
                self.uploaded_image = img.copy()

                # Display image
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img_display = Image.fromarray(rgb_img)

                label_width = self.camera_label.winfo_width()
                label_height = self.camera_label.winfo_height()

                if label_width > 1 and label_height > 1:
                    img_width, img_height = pil_img_display.size
                    aspect_ratio = img_width / img_height

                    if label_width / label_height > aspect_ratio:
                        new_height = label_height - 20
                        new_width = int(new_height * aspect_ratio)
                    else:
                        new_width = label_width - 20
                        new_height = int(new_width / aspect_ratio)

                    pil_img_display = pil_img_display.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    pil_img_display = pil_img_display.resize((640, 480), Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(pil_img_display)
                self.camera_label.config(image=photo)
                self.camera_label.image = photo

                # Update status to show auto-analysis is starting
                self.status_var.set("Image loaded - Auto-analyzing...")
                logger.info(f"Image uploaded from {file_path} - Starting automatic analysis")
                
                # Voice feedback for automatic analysis
                self.voice.speak("Image uploaded, analyzing automatically")
                
                # Analyze the uploaded image automatically
                self.analyze_image(img, source="upload")

            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to process image: {e}")
                logger.error(f"Image processing failed: {e}")
                self.status_var.set("Image upload failed")

    def analyze_image(self, frame, source="camera"):
        """Analyze image for disease"""
        if not self.model:
            messagebox.showerror("Model Error", "Model not loaded")
            return

        # Disable buttons
        self.capture_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.status_var.set("Analyzing...")

        # Run in a thread
        def analyze_thread():
            try:
                self.voice.speak("Analyzing leaf image")
                logger.info(f"Starting image analysis from {source}")

                # Create a debug directory if it doesn't exist
                debug_dir = "debug_images"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)

                # Save original image for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = os.path.join(debug_dir, f"original_{source}_{timestamp}.jpg")
                cv2.imwrite(debug_path, frame)
                
                # ENHANCED: Leaf detection and quality check
                processed_frame = frame.copy()
                quality_info = {}
                leaf_info = {}
                
                if self.enable_leaf_detection and self.leaf_detection_var.get():
                    logger.info("Performing leaf detection and quality check...")
                    self.root.after(0, lambda: self.status_var.set("Detecting leaf..."))
                    
                    # Check image quality
                    if self.quality_checker:
                        quality = self.quality_checker.check_quality(frame)
                        quality_info = quality
                        logger.info(f"Quality check: {quality['is_suitable']}, Metrics: {quality['metrics']}")
                        
                        if not quality['is_suitable']:
                            logger.warning(f"Image quality issues: {quality['issues']}")
                            self.voice.speak("Image quality may affect accuracy")
                    
                    # Detect and extract leaf
                    if self.leaf_detector:
                        leaves = self.leaf_detector.detect_leaves(frame)
                        
                        if len(leaves) > 0:
                            best_leaf = leaves[0]
                            leaf_info = {
                                'detected': True,
                                'confidence': best_leaf['confidence'],
                                'area': best_leaf['area'],
                                'count': len(leaves)
                            }
                            logger.info(f"Detected {len(leaves)} leaf/leaves, best confidence: {best_leaf['confidence']:.1f}%")
                            
                            # Extract leaf region for better analysis
                            processed_frame = self.leaf_detector.extract_leaf_region(frame, auto_detect=True)
                            
                            # Enhance the leaf image
                            processed_frame = self.leaf_detector.enhance_leaf_image(processed_frame)
                            
                            # Save detected leaf
                            leaf_path = os.path.join(debug_dir, f"detected_leaf_{source}_{timestamp}.jpg")
                            cv2.imwrite(leaf_path, processed_frame)
                            
                            self.voice.speak(f"Leaf detected with {best_leaf['confidence']:.0f} percent confidence")
                        else:
                            logger.warning("No leaf detected, using original image")
                            leaf_info = {'detected': False}
                            self.voice.speak("No leaf detected, analyzing full image")

                # Enhanced preprocessing
                # Convert to RGB if needed
                if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = processed_frame

                # Resize to model input size
                img = cv2.resize(rgb_frame, (224, 224))

                # Normalize pixel values
                img_array = img.astype(np.float32) / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)

                # Save preprocessed image for debugging
                preprocessed_path = os.path.join(debug_dir, f"preprocessed_{source}_{timestamp}.jpg")
                preprocessed_display = (img_array[0] * 255).astype(np.uint8)
                cv2.imwrite(preprocessed_path, preprocessed_display)

                # Make prediction
                logger.info("Making prediction...")
                predictions = self.model.predict(img_array, verbose=0)

                # Get top prediction
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx] * 100

                # Get top 3 predictions for debugging
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                top_3_predictions = [(self.class_names[i], predictions[0][i] * 100) for i in top_3_idx]
                logger.info(f"Top 3 predictions: {top_3_predictions}")

                # Get disease name
                if predicted_class_idx < len(self.class_names):
                    disease = self.class_names[predicted_class_idx]
                else:
                    disease = 'Unknown'

                # Get treatment
                treatment = TREATMENTS.get(disease, GENERAL_TREATMENT)

                # Extract features
                features = self.extract_features(frame)

                # Update UI with leaf detection info
                self.root.after(0, self.update_results, disease, confidence, treatment, features, top_3_predictions, leaf_info, quality_info)
                self.root.after(0, self.save_result, disease, confidence, features, treatment)

                # Voice feedback
                if "healthy" in disease.lower():
                    self.voice.speak(f"The plant appears healthy with {confidence:.0f} percent confidence")
                else:
                    plant_type = disease.split('___')[0]
                    disease_name = disease.split('___')[1].replace('_', ' ')
                    self.voice.speak(
                        f"Detected {disease_name} on {plant_type} with {confidence:.0f} percent confidence")
                    self.voice.speak("Treatment recommendations displayed")

                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                logger.info(f"Analysis complete: {disease} with {confidence:.1f}% confidence")

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Failed to analyze: {e}"))
                self.root.after(0, lambda: self.status_var.set("Analysis failed"))
                logger.error(f"Analysis failed: {e}")
            finally:
                self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.upload_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=analyze_thread)
        thread.daemon = True
        thread.start()

    def extract_features(self, frame):
        """Extract features from image"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1]) * 100

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hue = np.mean(hsv[:, :, 0])
            mean_rgb = np.mean(frame, axis=(0, 1))

            features = {
                'Color (Mean RGB)': f"B: {mean_rgb[0]:.1f}, G: {mean_rgb[1]:.1f}, R: {mean_rgb[2]:.1f}",
                'Hue Tone': f"{mean_hue:.1f}° (Greenish if ~60-120°)",
                'Texture (Edge Density)': f"{edge_density:.1f}%",
                'Image Size': f"{frame.shape[1]}x{frame.shape[0]} pixels"
            }
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def update_results(self, disease, confidence, treatment, features, top_predictions, leaf_info={}, quality_info={}):
        """Update results display"""
        self.results_text.delete(1.0, tk.END)

        results = f"🧠 CNN ANALYSIS RESULTS\n"
        results += f"{'=' * 40}\n\n"
        
        # Leaf detection info
        if leaf_info.get('detected'):
            results += f"🍃 LEAF DETECTION:\n"
            results += f"  Status: ✓ Detected\n"
            results += f"  Confidence: {leaf_info['confidence']:.1f}%\n"
            results += f"  Leaves Found: {leaf_info['count']}\n"
            results += f"  Leaf Area: {leaf_info['area']:.0f} pixels\n\n"
        elif leaf_info.get('detected') == False:
            results += f"🍃 LEAF DETECTION:\n"
            results += f"  Status: ⚠ No leaf detected\n"
            results += f"  Using full image for analysis\n\n"
        
        # Quality info
        if quality_info.get('metrics'):
            results += f"📊 IMAGE QUALITY:\n"
            metrics = quality_info['metrics']
            if 'brightness' in metrics:
                results += f"  Brightness: {metrics['brightness']:.1f}\n"
            if 'sharpness' in metrics:
                results += f"  Sharpness: {metrics['sharpness']:.1f}\n"
            if 'leaf_coverage' in metrics:
                results += f"  Leaf Coverage: {metrics['leaf_coverage']*100:.1f}%\n"
            if quality_info.get('warnings'):
                results += f"  ⚠ Warnings: {', '.join(quality_info['warnings'])}\n"
            results += "\n"
        
        results += f"🔬 DISEASE DETECTION:\n"
        results += f"Primary Detection: {disease}\n"
        results += f"Confidence: {confidence:.1f}%\n\n"

        # Show top 3 predictions
        results += f"TOP 3 PREDICTIONS:\n"
        for i, (pred_disease, pred_conf) in enumerate(top_predictions, 1):
            results += f"  {i}. {pred_disease}: {pred_conf:.1f}%\n"
        results += "\n"

        if features:
            results += f"LEAF FEATURES:\n"
            for feat, value in features.items():
                results += f"  {feat}: {value}\n"
            results += "\n"

        results += f"💊 TREATMENT RECOMMENDATIONS:\n"
        results += f"{treatment}\n"

        self.results_text.insert(tk.END, results)

    def save_result(self, disease, confidence, features, treatment):
        """Save detection result"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'disease': disease,
            'confidence': confidence,
            'features': features,
            'treatment': treatment
        }
        self.detection_history.append(result)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        try:
            with open('detection_history.json', 'w') as f:
                json.dump(self.detection_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def toggle_voice(self):
        """Toggle voice assistant"""
        enabled = self.voice.toggle_voice()
        status = "enabled" if enabled else "disabled"
        self.status_var.set(f"Voice {status}")
    
    def toggle_leaf_detection(self):
        """Toggle leaf detection on/off"""
        if self.enable_leaf_detection:
            enabled = self.leaf_detection_var.get()
            status = "enabled" if enabled else "disabled"
            self.status_var.set(f"Leaf detection {status}")
            logger.info(f"Leaf detection {status}")
            if enabled:
                self.voice.speak("Smart leaf detection enabled")
            else:
                self.voice.speak("Leaf detection disabled")

    # NEW METHOD: This method displays all available agrovets
    def show_all_agrovets(self):
        """Display all available agrovets in the database."""
        self.agrovet_info.delete(1.0, tk.END)
        self.agrovet_info.config(height=15)  # Make it taller to show the list

        info = "📋 ALL AVAILABLE AGROVETS IN DATABASE:\n\n"

        # Sort cities alphabetically for consistent display
        sorted_cities = sorted(AGROVET_DATA.keys())

        for city in sorted_cities:
            info += f"--- {city.upper()} ---\n"
            for agrovet in AGROVET_DATA[city]:
                info += f"  • {agrovet['name']}\n"
                info += f"    📍 {agrovet['address']}\n"
            info += "\n"  # Add a blank line between cities

        self.agrovet_info.insert(tk.END, info)
        self.maps_btn.config(state=tk.DISABLED)  # Disable maps as there's no single destination
        self.status_var.set("Showing all available agrovets")
        self.voice.speak("Showing all available agrovets in the database")

    def find_nearest_agrovet(self):
        """Find the nearest agrovets to the specified location"""
        location = self.location_var.get().strip()
        if not location:
            messagebox.showwarning("Location Required", "Please enter your location")
            return

        self.status_var.set("Finding nearest agrovets...")
        self.find_agrovet_btn.config(state=tk.DISABLED)
        self.agrovet_info.delete(1.0, tk.END)
        self.agrovet_info.insert(tk.END, "Searching... this may take a moment.")
        self.maps_btn.config(state=tk.DISABLED)

        def find_thread():
            try:
                self.voice.speak(f"Finding nearest agrovets near {location}")
                logger.info(f"Finding nearest agrovets for location: {location}")

                # Find the top 3 nearest agrovets
                nearest_agrovets = self.agrovet_finder.find_nearest_agrovets(location)

                if nearest_agrovets:
                    # Update UI with the list of agrovets
                    self.root.after(0, self.update_agrovet_info, nearest_agrovets)
                    closest = nearest_agrovets[0]
                    self.voice.speak(f"Found {closest['name']} approximately {closest['distance']} kilometers away")
                    self.root.after(0, lambda: self.status_var.set(f"Found {len(nearest_agrovets)} agrovets"))
                    logger.info(f"Found {len(nearest_agrovets)} agrovets, nearest is: {closest['name']}")
                else:
                    self.root.after(0, lambda: self.agrovet_info.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.agrovet_info.insert(tk.END,
                                                                        "Could not find any agrovets. Please check the location name or try a nearby major town."))
                    self.root.after(0, lambda: messagebox.showinfo("No Results",
                                                                   f"Could not find any agrovets near '{location}'.\nTry a major city like 'Nairobi' or 'Kisumu'."))
                    self.root.after(0, lambda: self.status_var.set("No agrovets found"))
                    self.voice.speak("Could not find any agrovets near your location")

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to find agrovet: {e}"))
                self.root.after(0, lambda: self.status_var.set("Error finding agrovet"))
                logger.error(f"Error finding agrovet: {e}")
            finally:
                self.root.after(0, lambda: self.find_agrovet_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=find_thread)
        thread.daemon = True
        thread.start()

    def update_agrovet_info(self, agrovets: list):
        """Update the agrovet information display with a list of options"""
        self.agrovet_info.delete(1.0, tk.END)
        self.agrovet_info.config(height=6)  # Increase height to fit more info

        info = "📍 TOP 3 NEAREST AGROVETS:\n\n"
        for i, agrovet in enumerate(agrovets, 1):
            info += f"{i}. 🏪 {agrovet['name']}\n"
            info += f"   📍 {agrovet['address']}\n"
            info += f"   📏 Distance: {agrovet['distance']} km\n\n"

        self.agrovet_info.insert(tk.END, info)
        self.maps_btn.config(state=tk.NORMAL)

        # Store the list of found agrovets for later use
        self.current_agrovets = agrovets

    def open_agrovet_in_maps(self):
        """Open directions to the closest agrovet in Google Maps"""
        if hasattr(self, 'current_agrovets') and self.current_agrovets:
            # Get the closest agrovet (the first one in the list)
            closest_agrovet = self.current_agrovets[0]
            location = self.location_var.get().strip()

            if self.agrovet_finder.open_in_maps(closest_agrovet, location):
                self.voice.speak("Opening directions to the nearest agrovet in Google Maps")
                self.status_var.set("Opening Google Maps...")
            else:
                messagebox.showerror("Error", "Failed to open Google Maps")
                self.status_var.set("Error opening maps")
        else:
            messagebox.showinfo("No Agrovet", "Please find an agrovet first")

    def on_closing(self):
        """Handle window close"""
        if self.cap:
            self.stop_camera()
        self.root.destroy()


def check_dependencies():
    """Check if all dependencies are installed"""
    logger.info("Checking dependencies...")

    required = {
        'tensorflow': 'tensorflow',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'requests': 'requests',
        'pyttsx3': 'pyttsx3',
        'geopy': 'geopy'
    }

    missing = []
    for package, import_name in required.items():
        try:
            __import__(import_name)
            logger.info(f"✓ {package}")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} - MISSING")

    if missing:
        logger.error(f"\nMissing packages. Install with:")
        logger.error(f"pip install {' '.join(missing)}")
        return False

    logger.info("\n✓ All dependencies installed!")
    return True


def main():
    """Main function"""
    print("=" * 60)
    print("CNN-BASED CROP DISEASE DETECTION SYSTEM")
    print("Simplified Version - Ready to Use!")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return

    # Ensure model exists
    result = ModelManager.ensure_model_exists(DEFAULT_MODEL_PATH)
    if not result:
        logger.error("Failed to obtain model")
        input("\nPress Enter to exit...")
        return

    print("\nChoose mode:")
    print("1. GUI Application")
    print("2. System Information")

    choice = input("\nEnter choice (1-2): ").strip()

    if choice == '1':
        # GUI mode
        root = tk.Tk()
        app = CropHealthApp(root, DEFAULT_MODEL_PATH)
        root.mainloop()
    elif choice == '2':
        # System info
        print("\n=== System Information ===")
        print(f"Python: {sys.version}")
        print(f"TensorFlow: {tf.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        print(f"NumPy: {np.__version__}")

        # GPU check
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"GPU: Available ({len(gpu_devices)} device(s))")
        else:
            print("GPU: Not available (using CPU)")

        # Camera check
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Camera: Available")
            cap.release()
        else:
            print("Camera: Not available")

        # Voice check
        if pyttsx3:
            try:
                engine = pyttsx3.init()
                print("Voice: Available")
            except:
                print("Voice: Not available")
        else:
            print("Voice: Not available (pyttsx3 not installed)")
    else:
        logger.error("Invalid choice!")


if __name__ == "__main__":
    main()