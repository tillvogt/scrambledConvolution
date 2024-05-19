import numpy as np

from .layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.max_matrix = None
        self.depth = None
        self.height = None
        self.width = None
        
    def forward(self, input):
        self.depth, self.height, self.width = input.shape
        output_shape = [self.depth, (self.height-self.pool_size)//self.stride +1, (self.width-self.pool_size)//self.stride +1]
        output = np.zeros(output_shape)
        self.max_matrix = np.zeros(output_shape)
        for d in range(output_shape[0]):
            for Y in range(output_shape[1]):
                for X in range(output_shape[2]):
                    y = Y*self.stride
                    x = X*self.stride
                    kernel = input[d, y:y+self.pool_size, x:x+self.pool_size]
                    self.max_matrix[d][Y][X] = np.argmax(kernel)
                    output[d][Y][X] = np.max(kernel)
                    
        return output
                    
        
                    
    def backward(self, output_gradient):
        input_gradient = np.zeros((self.depth, self.height, self.width))
        for d in range(output_gradient.shape[0]):
            for Y in range(output_gradient.shape[1]):
                for X in range(output_gradient.shape[2]):
                    _ = int(self.max_matrix[d][Y][X])
                    y = _//self.pool_size
                    x = _%self.pool_size
                    input_gradient[d][Y+y][X+x] = output_gradient[d][Y][X]
            
        return input_gradient, 0
