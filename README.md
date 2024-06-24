# Neural-Network-from-Scratch-on-Fashion-MNIST-Dataset

## Overview

This is a fun little attempt to learn the fundamentals of Deep Learning. I've built a basic neural network from scratch using Python and Numpy which contains custom-built components such as layers, optimizers, loss functions, and accuracy metrics. The dataset which I've used to train this model on is the Fashion MNIST Dataset. I initially aimed to train it on the MNIST Digit Recognizer Dataset but soon found out that it is comically easy to get higher accuracy and lower loss results on it (some say it is the "Hello World" equivalent of Deep Learning).

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## Resources

Everything that I needed to learn to make this comes from:

1. [3blue1brown](https://www.3blue1brown.com) (for fundamental maths concepts and a very good intro playlist into neural networks)
2. The [nnfs.io book](https://nnfs.io) by Harrison Kinsley (sentdex on YouTube) & Daniel Kukieła (this is a really awesome in-detail guide on how to build a neural network from scratch)

## Requirements

- Python

### Python Libraries

- OpenCV (cv2)
- NumPy

Install using pip:

```sh
pip install -r requirements.txt
```

## File Structure

- **main.py**: Defines the neural network model (`Model` class), layers (`Layer_Dense` class), activation functions (`Activation_ReLU`, `Activation_Softmax`), loss functions (`Loss`, `Loss_CategoricalCrossEntropy`), optimizers (`Optimizer_SGD`, `Optimizer_Adagrad`, `Optimizer_RMSprop`, `Optimizer_Adam`), and accuracy metrics (`Accuracy`, `Accuracy_Categorical`).
  
- **get_data.py**: To download the Fashion MNIST Dataset, extract it, and save the data under a folder called 'fashion_mnist_images' in the same directory.
  
- **load_data.py**: Functions to load Fashion MNIST dataset images into `X`, `y` (training data) and `X_val`, `y_val` (validation data) and then preprocess those values to get them ready for training.
  
- **train.py**: To train the model and save the optimized parameters and the model.
  
- **test.py**: To test the model for given images.

- **README.md**: This file, providing an overview and usage guide for the project.

- **requirements.txt**: Contains all the necessary python libraries to be installed with pip.

## Usage

1. **Get the Data**:

Run the `get_data.py` file.

2. **Load and Preprocess Dataset**:

Ensure the Fashion MNIST dataset images are in the `fashion_mnist_images` directory. Use `load_data.py` to load and preprocess the data.

3. **Configuring the Model**:

![Netowrk Architecture](https://github.com/Aditya-Dobhada/Neural-Network-from-Scratch-on-Fashion-MNIST-Dataset/assets/138972632/2fef494f-a7d7-4e10-839e-6ca0f2a4bd8a)


![Model Configuration](https://github.com/Aditya-Dobhada/Neural-Network-from-Scratch-on-Fashion-MNIST-Dataset/assets/138972632/a5a92c36-607b-44d8-adb3-647c068bde7e)


Define and configure the neural network model:

```python
# Initialize the neural network model
model = Model()

# Add layers to the model
model.add(Layer_Dense(X.shape[1], 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))  # Input layer to 1st hidden layer
model.add(Activation_ReLU())  # Activation function for the 1st hidden layer
model.add(Layer_Dense(128, 128))  # 1st hidden layer to 2nd hidden layer
model.add(Activation_ReLU())  # Activation function for the 2nd hidden layer
model.add(Layer_Dense(128, 10))  # 2nd hidden layer to output layer with 10 neurons (10 classes for Fashion MNIST)
model.add(Activation_Softmax())  # Softmax activation function for multiclass classification

# Set model parameters: loss function, optimizer, and accuracy metric
model.set(
    loss=Loss_CategoricalCrossEntropy(),  # Categorical cross-entropy loss for multiclass classification
    optimizer=Optimizer_Adam(learning_rate=0.002, decay=5e-4),  # Adam optimizer with specified learning rate and decay
    accuracy=Accuracy_Categorical()  # Accuracy metric for evaluation
)

# Finalize the model configuration after setting up layers, loss, optimizer, and accuracy
model.finalize()
```

```python
model.train(X, y, validation_data=(X_val, y_val), epochs=15, batch_size=128, print_every=100)
# Set the epoch value between 10-15, and let the other two values be as they are
```

4. **Train the Model**:

Run the `train.py` file. It also prints out the final evaluation stats of the model on the training and testing dataset. Then it saves the final trained parameters (weights and biases) and also saves the model object.

```python
# Evaluate the trained model on validation data (X_val, y_val)
print('\nThe final evaluation of the model on the training dataset is:')
model.evaluate(X, y)
print('\nThe final evaluation of the model on the testing dataset is:')
model.evaluate(X_val, y_val)

# Save trained model parameters to a file named 'fashion_mnist.parms'
model.save_parameters('fashion_mnist.parms')

# Save the entire model (architecture, weights, etc.) to a file named 'fashion_mnist.model'
model.save('fashion_mnist.model')
```

5. **Test the Model**:

`test.py` file loads the saved model. Then we pass the path to the image we wish to test for.

```python
# Load the trained model from file
model = Model.load('fashion_mnist.model')

# Preprocess an image file (for eg, 'tshirt.png') to prepare it for prediction
image_data = preprocess_image('tshirt.png')
```

Run the `test.py` file. The prediction is printed according to the labels declared above.

## Fashion MNIST Labels

The model predicts the following categories of fashion items:

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

Ensure the `fashion_mnist_labels` dictionary in `test.py` matches these labels for correct predictions.

## Evaluation Stats

The accuracy achieved on the default state of this code:
![Evaluation](https://github.com/Aditya-Dobhada/Neural-Network-from-Scratch-on-Fashion-MNIST-Dataset/assets/138972632/f7731c6b-22b0-408b-8303-310c5b1b9a78)

This translates to 90.5% accuracy and 25.6% loss on the training data, and 87.5% accuracy and 34.2% loss on the testing data. A pretty decent result with room for improvement.
