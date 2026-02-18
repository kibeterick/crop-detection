# -*- coding: utf-8 -*-
"""
Configuration Management for Crop Disease Detection System
"""
import os
import json
from typing import Dict, Any

class Config:
    """Configuration manager"""
    
    # Paths
    DEFAULT_MODEL_PATH = 'crop_disease_cnn_model.keras'
    DEMO_MODEL_PATH = 'demo_model.keras'
    LOGS_DIR = 'logs'
    DEBUG_DIR = 'debug_images'
    HISTORY_FILE = 'detection_history.json'
    CONFIG_FILE = 'app_config.json'
    
    # Model settings
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MAX_HISTORY = 50
    
    # UI settings
    WINDOW_SIZE = "1200x850"
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Voice settings
    VOICE_RATE = 150
    VOICE_VOLUME = 0.9
    
    # Agrovet settings
    MAX_AGROVET_RESULTS = 5
    GEOCODE_TIMEOUT = 10
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(Config.CONFIG_FILE):
            try:
                with open(Config.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(Config.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass
    
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist"""
        for directory in [Config.LOGS_DIR, Config.DEBUG_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
