# -*- coding: utf-8 -*-
"""
Model Training Utilities for Crop Disease Detection
"""
import os
import logging
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import Config
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train custom CNN models for crop disease detection"""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing training data (with subdirectories for each class)
            img_size: Input image size
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
        # Configure GPU
        self.configure_gpu()
    
    def configure_gpu(self):
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configured: {len(gpus)} device(s)")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.info("No GPU found, using CPU")
    
    def create_data_generators(self, validation_split: float = 0.2):
        """
        Create data generators with augmentation
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        return train_generator, validation_generator
    
    def build_model(self, num_classes: int, architecture: str = 'simple'):
        """
        Build CNN model
        
        Args:
            num_classes: Number of output classes
            architecture: Model architecture ('simple', 'mobilenet', 'resnet')
        """
        if architecture == 'simple':
            self.model = self._build_simple_cnn(num_classes)
        elif architecture == 'mobilenet':
            self.model = self._build_mobilenet(num_classes)
        elif architecture == 'resnet':
            self.model = self._build_resnet(num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        logger.info(f"Model built: {architecture}")
        return self.model
    
    def _build_simple_cnn(self, num_classes: int):
        """Build a simple CNN model"""
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),
            
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_mobilenet(self, num_classes: int):
        """Build MobileNetV2-based model"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_resnet(self, num_classes: int):
        """Build ResNet50-based model"""
        base_model = tf.keras.applications.ResNet50(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs: int = 20, save_path: str = 'trained_model.keras'):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            save_path: Path to save the trained model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Callbacks
        callback_list = [
            callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
        
        # Train
        logger.info("Starting training...")
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info(f"Training complete. Model saved to {save_path}")
        return self.history
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
        plt.close()
    
    def evaluate(self, test_dir: str):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4f}")
        
        return results


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(data_dir='data/train', img_size=(224, 224), batch_size=32)
    
    # Build model
    trainer.build_model(num_classes=8, architecture='simple')
    
    # Train
    trainer.train(epochs=20, save_path='my_custom_model.keras')
    
    # Plot history
    trainer.plot_training_history('my_training_history.png')
    
    # Evaluate
    trainer.evaluate('data/test')
