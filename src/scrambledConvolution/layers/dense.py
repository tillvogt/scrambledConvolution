import numpy as np
from .layer import Layer

class Dense(Layer):
    """
    Dense (Fully Connected) Layer to perform affine transformation and provide backward pass functionality.
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the dense layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Perform the forward pass of the dense layer.

        Args:
            input (numpy.ndarray): Input data of shape (input_size, batch_size).

        Returns:
            numpy.ndarray: Output data after applying the affine transformation, of shape (output_size, batch_size).
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        """
        Perform the backward pass of the dense layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss with respect to the output, of shape (output_size, batch_size).

        Returns:
            tuple: Gradients with respect to the input and weights.
                - input_gradient (numpy.ndarray): Gradient with respect to the input, of shape (input_size, batch_size).
                - weights_gradient (numpy.ndarray): Gradient with respect to the weights, of shape (output_size, input_size).
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient, weights_gradient
    
    def learning(self, weights_grad, bias_grad):
        """
        Update the weights and biases of the dense layer.

        Args:
            weights_grad (numpy.ndarray): Gradient of the loss with respect to the weights.
            bias_grad (numpy.ndarray): Gradient of the loss with respect to the biases.
        """
        self.weights -= weights_grad
        self.bias -= bias_grad
        