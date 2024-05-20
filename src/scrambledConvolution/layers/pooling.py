import numpy as np
from .layer import Layer

def unravel(index, upper_bound):
    """
    Convert a flat index into a 2D index within the bounds of the upper bound.

    Args:
        index (int): The flat index to be unraveled.
        upper_bound (int): The upper bound for the 2D index.

    Returns:
        tuple: A tuple (y, x) representing the 2D index.
    """
    index = int(index)
    y = index // upper_bound
    x = index % upper_bound
    
    return y, x

def maxpooling(input, output, output_shape, max_matrix, pool_size, stride):
    """
    Perform max pooling on the input.

    Args:
        input (numpy.ndarray): Input data of shape (depth, height, width).
        output (numpy.ndarray): Output data of shape (depth, pooled_height, pooled_width).
        output_shape (list): Shape of the output data as [depth, pooled_height, pooled_width].
        max_matrix (numpy.ndarray): Matrix to store the indices of the maximum values.
        pool_size (int): Size of the pooling window.
        stride (int): Stride of the pooling window.
    """
    for d in range(output_shape[0]):
        for Y in range(output_shape[1]):
            for X in range(output_shape[2]):
                y = Y * stride
                x = X * stride
                
                pooling_kernel = input[d, y:y + pool_size, x:x + pool_size]
                max_idx = pooling_kernel.argmax()
                max_y, max_x = unravel(max_idx, pool_size)
                
                max_matrix[d][Y][X] = max_idx
                output[d][Y][X] = pooling_kernel[max_y, max_x]
                
class Pooling(Layer):
    """
    Pooling Layer to perform max pooling and provide backward pass functionality.
    """
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize the pooling layer.

        Args:
            pool_size (int, optional): Size of the pooling window. Default is 2.
            stride (int, optional): Stride of the pooling window. Default is 2.
        """
        self.pool_size = pool_size
        self.stride = stride
        self.max_matrix = None
        self.depth = None
        self.height = None
        self.width = None
        
    def forward(self, input):
        """
        Perform the forward pass of the pooling layer.

        Args:
            input (numpy.ndarray): Input data of shape (depth, height, width).

        Returns:
            numpy.ndarray: Output data after applying max pooling, of shape (depth, pooled_height, pooled_width).
        """
        self.depth, self.height, self.width = input.shape
        output_shape = [self.depth, (self.height - self.pool_size) // self.stride + 1, (self.width - self.pool_size) // self.stride + 1]
        output = np.zeros(output_shape)
        self.max_matrix = np.zeros(output_shape)
        
        maxpooling(input, output, output_shape, self.max_matrix, self.pool_size, self.stride)
        
        return output
                    
    def backward(self, output_gradient):
        """
        Perform the backward pass of the pooling layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss with respect to the output, of shape (depth, pooled_height, pooled_width).

        Returns:
            tuple: Gradients with respect to the input and zero (since pooling has no learnable parameters).
                - input_gradient (numpy.ndarray): Gradient with respect to the input, of shape (depth, height, width).
                - int: Always returns 0 as there are no learnable parameters in pooling.
        """
        input_gradient = np.zeros((self.depth, self.height, self.width))
        for d in range(output_gradient.shape[0]):
            for Y in range(output_gradient.shape[1]):
                for X in range(output_gradient.shape[2]):
                    y, x = unravel(int(self.max_matrix[d][Y][X]), self.pool_size)
                    input_gradient[d][Y * self.stride + y][X * self.stride + x] = output_gradient[d][Y][X]
            
        return input_gradient, 0
    