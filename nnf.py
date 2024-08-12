import numpy as np
import pickle

class Model:
    def __init__(self):
        # This sets up an empty list for layers. We start with no loss function or optimizer.
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add_layer(self, layer):
        # Here, we’re just adding a new layer to the list. The layers are what make up our neural network.
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        # This is where we set up the loss function and optimizer. They are crucial for training the model.
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        # This function runs a forward pass through all the layers. 
        # Basically, it processes the input data through each layer of the network.
        self.inputs = [x]  # Keep track of inputs for backward pass later
        for layer in self.layers:
            x = layer.forward(x)  # Pass data through each layer's forward method
            self.inputs.append(x)  # Save the output of each layer
        return x  # The final output of the model

    def backward(self, gradient):
        # This function handles the backward pass where gradients are calculated.
        # It works backwards through the layers to compute how each layer's weights should be updated.
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)  # Update gradients for each layer

    def train(self, x_train, y_train, epochs, batch_size):
        num_samples = x_train.shape[0]  # Total number of samples in the training data
        for epoch in range(epochs):
            epoch_loss = 0  # Keep track of the loss for this epoch
            for i in range(0, num_samples, batch_size):
                # Get a batch of data from the training set
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Forward pass: Get the predictions from the model
                predictions = self.forward(x_batch)
                
                # Compute the loss and gradients
                loss = self.loss.forward(predictions, y_batch)
                epoch_loss += loss
                gradient = self.loss.backward(predictions, y_batch)
                
                # Backward pass: Calculate gradients for each layer
                self.backward(gradient)
                
                # Update the parameters using the optimizer
                self.optimizer.step(self.layers)
                
            # Average loss over the epoch (since we processed batches)
            epoch_loss /= (num_samples / batch_size)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

    def evaluate(self, x_test, y_test):
        # Run the model on test data to see how well it performs
        predictions = self.forward(x_test)  # Get predictions
        loss = self.loss.forward(predictions, y_test)  # Calculate loss
        accuracy = self.compute_accuracy(predictions, y_test)  # Calculate accuracy
        return loss, accuracy

    def predict(self, x):
        # Simply run a forward pass to get predictions
        return self.forward(x)

    def compute_accuracy(self, predictions, labels):
        # Compare the predicted classes to the true labels to compute accuracy
        predicted_classes = predictions.argmax(axis=1)  # Get the class with the highest probability
        true_classes = labels.argmax(axis=1)  # Get the true class labels
        accuracy = np.mean(predicted_classes == true_classes)  # Calculate the percentage of correct predictions
        return accuracy

    def save(self, filename):
        # Save the model's layers, optimizer, and loss function to a file
        with open(filename, 'wb') as file:
            pickle.dump({
                'layers': self.layers,
                'optimizer': self.optimizer,
                'loss': self.loss
            }, file)

    def load(self, filename):
        # Load the model's saved layers, optimizer, and loss function from a file
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.layers = data['layers']
            self.optimizer = data['optimizer']
            self.loss = data['loss']


class SGD:
    def __init__(self, learning_rate):
        # Initialize the optimizer with a learning rate
        # Learning rate determines how big each step is during optimization
        self.learning_rate = learning_rate

    def step(self, layers):
        # Update weights and biases for all layers
        for layer in layers:
            if hasattr(layer, 'weights'):
                # Check if the layer has weights
                # Adjust weights and biases based on their gradients and learning rate
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias

class MSELoss:
    def forward(self, predictions, labels):
        # Compute Mean Squared Error (MSE) loss
        # Predictions: model's output
        # Labels: true values
        # Loss = average of squared differences between predictions and true labels
        self.predictions = predictions
        self.labels = labels
        return np.mean((predictions - labels) ** 2)

    def backward(self, predictions, labels):
        # Compute gradient of the MSE loss with respect to predictions
        # Gradient = 2 times the difference between predictions and labels, normalized by the number of samples
        return 2 * (predictions - labels) / predictions.size

class CrossEntropyLoss:
    def forward(self, predictions, labels):
        # Compute Cross-Entropy Loss
        # Predictions: model's output (probabilities)
        # Labels: true class labels (one-hot encoded)
        # Clip predictions to avoid log(0) issues and calculate the loss
        self.predictions = np.clip(predictions, 1e-9, 1. - 1e-9)
        self.labels = labels
        loss = -np.sum(labels * np.log(self.predictions)) / predictions.shape[0]
        return loss

    def backward(self, predictions, labels):
        # Compute gradient of the Cross-Entropy loss with respect to predictions
        # Gradient = difference between predictions and labels, normalized by the number of samples
        return (predictions - labels) / predictions.shape[0]

class Softmax:
    def forward(self, x):
        # Compute the Softmax function
        # Softmax converts raw scores into probabilities (values between 0 and 1 that sum to 1)
        # First, subtract the max value from x for numerical stability (avoids overflow)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        # Compute softmax by dividing exponentiated values by the sum of exps across classes
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, gradient):
        # For softmax, the backward pass doesn't change the gradient
        # Usually, it's combined with the cross-entropy loss gradient in practice
        return gradient

class ReLU:
    def forward(self, x):
        # Apply ReLU activation function
        # ReLU replaces negative values with zero, keeping positive values as they are
        self.input = x
        return np.maximum(0, x)

    def backward(self, gradient):
        # For ReLU, the gradient is propagated only for positive inputs
        # If the input was positive, pass the gradient through, otherwise pass zero
        return gradient * (self.input > 0)

class Linear:
    def __init__(self, input_size, output_size):
        # Initialize a fully connected (dense) layer with random weights and zero biases
        # Weights are small random numbers, biases are zeros
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        # Compute the output of the linear layer
        # Perform a matrix multiplication of input x with weights and add biases
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, gradient):
        # Compute gradients of weights and biases
        # Gradient with respect to weights is input transposed times the gradient
        self.grad_weights = np.dot(self.input.T, gradient)
        # Gradient with respect to biases is the sum of gradients
        self.grad_bias = np.sum(gradient, axis=0, keepdims=True)
        # Return the gradient with respect to inputs for the previous layer
        return np.dot(gradient, self.weights.T)
