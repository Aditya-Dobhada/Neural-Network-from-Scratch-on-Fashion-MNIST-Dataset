# Importing everything from main.py 
from main import *

# Dictionary mapping Fashion MNIST labels to their corresponding classes
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Load the trained model from file 
model = Model.load('fashion_mnist.model')

# Preprocess an image file (for eg, 'tshirt.png') to prepare it for prediction
image_data = preprocess_image('tshirt.png')

# Use the loaded model to make predictions on the preprocessed image data
confidences = model.predict(image_data)

# Convert confidence scores into class predictions using softmax activation
predictions = model.output_layer_activation.predictions(confidences)

# Retrieve the predicted class label from the fashion_mnist_labels dictionary


for prediction in predictions:
    output = fashion_mnist_labels[prediction]
    print(output)
# Print the predicted label
