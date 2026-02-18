import tensorflow as tf
try:
    import tensorflow.keras as keras
    keras_version = tf.keras.__version__ if hasattr(tf.keras, '__version__') else keras.__version__
except AttributeError:
    import keras
    keras_version = keras.__version__

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras_version}")
print(f"Physical devices: {tf.config.list_physical_devices('GPU')}")
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Keras is working!")