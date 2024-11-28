import os
import numpy as np
import tensorflow as tf
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def load_fits_data(data_dir, image_size=(128, 128)):
    """
    Load .fits images and labels from a directory.
    Args:
        data_dir (str): Directory containing .fits files.
        image_size (tuple): Desired size of the images (width, height).
    Returns:
        X (numpy.ndarray): Array of images.
        y (numpy.ndarray): Array of labels (0 for no_satellite, 1 for with_satellite).
    """
    X, y = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".fits"):
            # Load FITS image
            fits_path = os.path.join(data_dir, file_name)
            with fits.open(fits_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
            
            # Normalize image
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))  # Normalize to [0, 1]
            
            # Add an extra dimension to make it a 3D array
            image_data = np.expand_dims(image_data, axis=-1)
            
            # Resize image
            image_resized = tf.image.resize(image_data, image_size).numpy()
            X.append(image_resized)

            # Extract label from file name
            if "no_satellite" in file_name:
                y.append(0)
            elif "with_satellite" in file_name:
                y.append(1)

    X = np.array(X)  # Shape will be (num_images, height, width, 1)
    y = np.array(y)
    return X, y