import numpy as np
import pickle
import copy
import cv2

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., weight_regularizer_l2=0.,
                 bias_regularizer_l1=0., bias_regularizer_l2=0.):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Initialize regularization parameters
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs):
        # Store inputs for later use in backpropagation
        self.inputs = inputs
        # Perform forward pass calculation: output = inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Calculate gradients with respect to weights, biases, and inputs
        
        # Gradient of weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # Gradient of biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Regularization for weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # Regularization for biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient with respect to inputs for backpropagation to previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def get_parameters(self):
        # Return current weights and biases
        return self.weights, self.biases    

    def set_parameters(self, weights, biases):
        # Set new weights and biases
        self.weights = weights
        self.biases = biases
  

# Input Layer  
class Layer_Input:
    # Represents the input layer of the neural network.
    
    def forward(self, inputs):
        # Set the output of the input layer to the inputs themselves
        self.output = inputs
  
 
# ReLU Activation     
class Activation_ReLU:
    def forward(self, inputs):
        # Store inputs for later use in backpropagation
        self.inputs = inputs
        # Apply ReLU activation function element-wise
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        # Copy dvalues since we will modify it
        self.dinputs = dvalues.copy()
        # Zero out gradients where input values were <= 0
        self.dinputs[self.inputs <= 0] = 0
        

# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        # Calculate softmax probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        # Initialize array to store gradients
        self.dinputs = np.empty_like(dvalues)
        
        # Iterate over outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate gradient and add it to the array of gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    def predictions(self, outputs):
        # Return predictions as the index of the highest probability in each output vector
        return np.argmax(outputs, axis=1)


# Common loss class
class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        
        # Calculate regularization loss for each trainable layer
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
                
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
    def remember_trainable_layers(self, trainable_layers):
        # Remember the layers that have trainable parameters
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, *, include_regularization=False):
        # Calculate data loss and optionally regularization loss
        
        sample_losses = self.forward(output, y)  # Calculate loss for each sample #type: ignore
        data_loss = np.mean(sample_losses)  # Compute mean loss across all samples
        
        # Accumulate sum and count of losses for statistics
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_regularization:
            return data_loss  # Return only data loss if regularization is not included
        
        return data_loss, self.regularization_loss()  # Return both data loss and regularization loss
    
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate accumulated data loss and optionally accumulated regularization loss
        
        data_loss = self.accumulated_sum / self.accumulated_count  # Compute mean accumulated data loss
        
        if not include_regularization:
            return data_loss  # Return only accumulated data loss if regularization is not included
        
        return data_loss, self.regularization_loss()  # Return both accumulated data loss and regularization loss
    
    def new_pass(self):
        # Reset accumulated sum and count for a new pass (epoch)
        self.accumulated_sum = 0
        self.accumulated_count = 0
        
 
# Cross Entropy Loss       
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Calculate categorical cross-entropy loss
        
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predicted values to avoid log(0)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)  # Compute negative log likelihoods
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Calculate gradients of categorical cross-entropy loss
        
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]  # Convert y_true to one-hot encoded if it's not
        
        self.dinputs = - y_true / dvalues  # Calculate gradients
        self.dinputs = self.dinputs / samples  # Average gradients over samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self, dvalues, y_true):
        # Backward pass through softmax and categorical cross-entropy combined
        
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoded y_true to class indices
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1  # Calculate gradient of softmax and cross-entropy
        self.dinputs = self.dinputs / samples  # Average gradients over samples


# SGD Optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        # Update learning rate based on decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        # Update weights and biases using SGD with optional momentum
        
        if self.momentum:
            # Initialize momentums if not already initialized
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            # Update momentums
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            # Standard SGD update without momentum
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # Update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        # Increment iteration count after each update
        self.iterations += 1


# Adagrad Optimizer
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    
    def pre_update_params(self):
        # Update learning rate based on decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        # Update weights and biases using Adagrad
        
        # Initialize caches if not already initialized
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update caches
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        # Update weights and biases
        layer.weights += -self.current_learning_rate * layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
                                   
    def post_update_params(self):
        # Increment iteration count after each update
        self.iterations += 1


# RMSprop Optimizer
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def pre_update_params(self):
        # Update learning rate based on decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        # Update weights and biases using RMSprop
        
        # Initialize caches if not already initialized
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update caches using RMSprop formula
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2
        
        # Update weights and biases
        layer.weights += -self.current_learning_rate * layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
                                   
    def post_update_params(self):
        # Increment iteration count after each update
        self.iterations += 1


# Adam Optimizer
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def pre_update_params(self):
        # Update learning rate based on decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        # Update weights and biases using Adam optimizer
        
        # Initialize momentums and caches if not already initialized
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentums with bias correction
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
        
        # Bias correction for momentums
        weight_momentums_corrected = layer.weight_momentums / \
                        (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                        (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update caches with bias correction
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        
        # Bias correction for caches
        weight_cache_corrected = layer.weight_cache / \
                        (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                        (1 - self.beta_2 ** (self.iterations + 1))
        
        # Update weights and biases
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
                                   
    def post_update_params(self):
        # Increment iteration count after each update
        self.iterations += 1


# Model class
class Model:
    
    def __init__(self):
        self.layers = []  # List to store all layers in the model
        self.softmax_classifier_output = None  # Placeholder for softmax classifier output
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        # Set loss, optimizer, and accuracy for the model
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
   
    def finalize(self):
        # Finalize the model architecture
        
        self.input_layer = Layer_Input()  # Initialize input layer
        
        layer_count = len(self.layers)
        self.trainable_layers = []  # List to store trainable layers
        
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
 
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Train the model
        
        self.accuracy.init(y)
        train_steps = 1
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                    
                output = self.forward(batch_X)
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                self.backward(output, batch_y)
                
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')
            
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')
            
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
  
    def forward(self, X):
        # Perform forward propagation
        
        self.input_layer.forward(X)
        
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output
  
    def backward(self, output, y):
        # Perform backward propagation
        
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs 
            
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return
        
        self.loss.backward(output, y)
        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Evaluate the model on validation data
        
        validation_steps = 1
        
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
                
            output = self.forward(batch_X)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        print(f'validation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')

    def get_parameters(self):
        # Get parameters of all trainable layers
        
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    def set_parameters(self, parameters):
        # Set parameters of all trainable layers
        
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameters(self, path):
        # Save parameters of the model to a file
        
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
        print('Parameters saved.')
      
    def load_parameters(self, path):
        # Load parameters into the model from a file
        
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
        print('Parameters loaded.')

    def save(self, path):
        # Save the entire model (including architecture and parameters) to a file
        
        model = copy.deepcopy(self)
        
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
                
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        print('Model saved.')

    @staticmethod
    def load(path):
        # Load the entire model from a file
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print('Model loaded.')   
        return model

    def predict(self, X, *, batch_size=None):
        # Make predictions on input data
        
        prediction_steps = 1
        
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            batch_output = self.forward(batch_X)
            output.append(batch_output)
        
        return np.vstack(output)


# Common Accuracy class       
class Accuracy:
    # Base class for computing accuracy metrics.
    
    def calculate(self, predictions, y):
        # Compare predictions with ground truth y and compute accuracy
        
        comparisons = self.compare(predictions, y)  # Call compare method to get comparisons #type: ignore
        
        accuracy = np.mean(comparisons)  # Compute mean accuracy
        
        # Accumulate sum of correct predictions and total count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy
    
    def calculate_accumulated(self):
        # Compute accumulated accuracy
        
        accuracy = self.accumulated_sum / self.accumulated_count  # Calculate average accuracy
        
        return accuracy
    
    def new_pass(self):
        # Reset accumulated sums and counts for a new pass
        
        self.accumulated_sum = 0
        self.accumulated_count = 0
  
# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    # Accuracy calculation class for categorical data.
    
    def init(self, y):
        # Initialization method for categorical accuracy (no specific initialization required)
        pass
    
    def compare(self, predictions, y):
        # Compare predictions with ground truth labels y
        
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)  # If y is one-hot encoded, convert it back to class labels
        
        # Return an array of booleans indicating whether predictions match y
        return predictions == y
   


# Image_preprocessing
def preprocess_image(path):
    # Read the image in grayscale
    image_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was successfully read
    if image_data is None:
        raise FileNotFoundError(f"Image file '{path}' not found.")
    
    # Resize the image to 28x28 pixels
    image_data = cv2.resize(image_data, (28, 28))
    
    # Invert the image (convert to negative)
    image_data = 255 - image_data
    
    # Reshape and normalize the image
    image_data = (image_data.reshape(1, -1).astype(np.float32)) / 255
    
    # Return the preprocessed image data (transposed)
    return image_data
