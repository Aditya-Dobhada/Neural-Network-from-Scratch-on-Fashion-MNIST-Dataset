import os
import cv2
import numpy as np


def load_mnist_dataset(dataset, path):
    # Get list of labels (subdirectories) in the specified dataset directory
    labels = os.listdir(os.path.join(path, dataset))

    X = []  # List to store images
    y = []  # List to store labels (class names)

    for label in labels:
        label_path = os.path.join(path, dataset, label)
        
        if not os.path.isdir(label_path):  # Skip if it's not a directory
            continue
        
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            
            if not os.path.isfile(file_path):  # Skip if it's not a file
                continue
            
            if file == ".DS_Store":  # Skip .DS_Store file if present (Mac specific)
                continue
            
            # Read image in its original format (unchanged)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            X.append(image)  # Append image data to X
            y.append(label)   # Append label to y

    return np.array(X), np.array(y).astype('uint8')  # Convert lists to numpy arrays and return


def create_data_mnist(path):
    # Load training and testing datasets
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test


# Load Fashion MNIST dataset from specified path
X, y, X_val, y_val = create_data_mnist('fashion_mnist_images')


# Shuffle data using random keys
keys = np.array(range(X.shape[0]))  # Create an array of indices for shuffling
np.random.shuffle(keys)  # Shuffle the indices randomly

X = X[keys]  # Shuffle X data based on shuffled keys
y = y[keys]  # Shuffle y data based on shuffled keys


# Normalize pixel values to the range [0, 1] and flatten images
X = (X.reshape(X.shape[0], -1).astype(np.float32)) / 255
X_val = (X_val.reshape(X_val.shape[0], -1).astype(np.float32)) / 255

