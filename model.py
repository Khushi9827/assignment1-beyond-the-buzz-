import numpy as np
import pandas as pd

class ANN:
    def __init__(self, input_size, output_size, learning_rate, num_hidden_layers, num_nodes_hidden):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_hidden = num_nodes_hidden
        
        self.parameters = {}  # Dictionary to store the model parameters (weights and biases)
        
    def _initialize_parameters(self):
        np.random.seed(42)  # For reproducibility
        
        # Initialize weights and biases for each hidden layer
        input_dim = self.input_size
        for layer in range(1, self.num_hidden_layers + 1):
            output_dim = self.num_nodes_hidden
            self.parameters['W' + str(layer)] = np.random.randn(output_dim, input_dim) * 0.01
            self.parameters['b' + str(layer)] = np.zeros((output_dim, 1))
            input_dim = output_dim
        
        # Initialize weights and biases for the output layer
        self.parameters['W' + str(self.num_hidden_layers + 1)] = np.random.randn(self.output_size, input_dim) * 0.01
        self.parameters['b' + str(self.num_hidden_layers + 1)] = np.zeros((self.output_size, 1))
        
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def _softmax(self, Z):
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def _forward_propagation(self, X):
        cache = {'A0': X}  # Cache to store the activations
        
        for layer in range(1, self.num_hidden_layers + 1):
            A_prev = cache['A' + str(layer - 1)]
            W = self.parameters['W' + str(layer)]
            b = self.parameters['b' + str(layer)]
            
            Z = np.dot(W, A_prev) + b
            A = self._sigmoid(Z)
            
            cache['Z' + str(layer)] = Z
            cache['A' + str(layer)] = A
        
        A_prev = cache['A' + str(self.num_hidden_layers)]
        W = self.parameters['W' + str(self.num_hidden_layers + 1)]
        b = self.parameters['b' + str(self.num_hidden_layers + 1)]
        
        Z = np.dot(W, A_prev) + b
        A = self._softmax(Z)
        
        cache['Z' + str(self.num_hidden_layers + 1)] = Z
        cache['A' + str(self.num_hidden_layers + 1)] = A
        
        return A, cache
    
    def _compute_loss(self, A, Y):
        m = Y.shape[1]  # Number of examples
        
        # Calculate cross-entropy loss
        loss = -np.sum(Y * np.log(A + 1e-8)) / m
        
        return loss
    
    def _backward_propagation(self, X, Y, cache):
        grads = {}  # Dictionary to store the gradients
        
        m = X.shape[1]  # Number of examples
        
        dZ = cache['A' + str(self.num_hidden_layers + 1)] - Y
        dW = np.dot(dZ, cache['A' + str(self.num_hidden_layers)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        grads['dW' + str(self.num_hidden_layers + 1)] = dW
        grads['db' + str(self.num_hidden_layers + 1)] = db
        
        for layer in reversed(range(1, self.num_hidden_layers + 1)):
            dA = np.dot(self.parameters['W' + str(layer + 1)].T, dZ)
            dZ = dA * cache['A' + str(layer)] * (1 - cache['A' + str(layer)])
            dW = np.dot(dZ, cache['A' + str(layer - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            grads['dW' + str(layer)] = dW
            grads['db' + str(layer)] = db
        
        return grads
    
    def train(self, X, Y, num_iterations):
        self._initialize_parameters()
        
        for i in range(num_iterations):
            A, cache = self._forward_propagation(X)
            loss = self._compute_loss(A, Y)
            grads = self._backward_propagation(X, Y, cache)
            
            # Update parameters
            for layer in range(1, self.num_hidden_layers + 2):
                self.parameters['W' + str(layer)] -= self.learning_rate * grads['dW' + str(layer)]
                self.parameters['b' + str(layer)] -= self.learning_rate * grads['db' + str(layer)]
            
            if i % 100 == 0:
                print(f"Loss after iteration {i}: {loss}")
    
    def predict(self, X):
        A, _ = self._forward_propagation(X)
        predictions = np.argmax(A, axis=0)
        return predictions
