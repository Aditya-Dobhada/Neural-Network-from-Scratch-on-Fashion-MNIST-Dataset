from load_data import X, y, X_val, y_val  # Importing training and validation data from load_data module
from main import *  # Importing everything from main.py 


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

# Train the model on training data (X, y) with validation on validation data (X_val, y_val)
model.train(X, y, validation_data=(X_val, y_val), epochs=15, batch_size=128, print_every=100)

# Evaluate the trained model on validation data (X_val, y_val)
print('\n')
print('The final evaluation of the model on the training dataset is:')
model.evaluate(X, y)
print('The final evaluation of the model on the testing dataset is:')
model.evaluate(X_val, y_val)

# Save trained model parameters to a file named 'fashion_mnist.parms'
model.save_parameters('fashion_mnist.parms')

# Save the entire model (architecture, weights, etc.) to a file named 'fashion_mnist.model'
model.save('fashion_mnist.model')

