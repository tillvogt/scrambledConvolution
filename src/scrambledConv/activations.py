import numpy as np

from .layers import Layer

class Tanh(Layer):
    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient):
        factor = 1- self.output ** 2
        return np.multiply(output_gradient, factor), 0
    
     
class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
        
    def backward(self, output_gradient):
        factor = self.output * (1-self.output)
        return np.multiply(output_gradient, factor), 0
    
        
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient), 0   
