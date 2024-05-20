import numpy as np
from numba import njit
from .layer import Layer


"""
In order to ensure the accurate implementation of Backpropagation despite variations in the mathematical procedure, 
all coefficients are stored in the 'correlation matrix'. 
Consequently, each cross-correlation represents a system of linear equations with numerous terms that are equal to zero.
"""

def corrmatrix(input_height, input_width, kernel_size):
    """
    Generates a 4D correlation matrix for cross-correlation operations.

    Args:
    input_height (int): height of the inputarray in pixels
    input_width (int): width of the inputarray in pixels
    kernel_size (int): sidelength of the correlation kernel in pixels 
    
    Returns:
    corrmat (numpy.ndarray): The 4D correlation matrix. First two dimensions are equal to the outputdimension (input-kernelsize+1)
    Second two are equal to the inputdimension.
    corrmat_shape (list(int)): The shape of the 4D correlation matrix.
    """
    
    corrmat_shape = [input_height-kernel_size+1, input_width-kernel_size+1, input_height, input_width]
    corrmat = np.zeros(corrmat_shape).astype(np.uint8)
    # Populate the correlation matrix based on the kernel size
    for Y in range(corrmat_shape[0]):
        for X in range(corrmat_shape[1]):
            for y in range(kernel_size):
                for x in range(kernel_size):
                    corrmat[Y, X, Y+y, X+x] = (y*kernel_size)+x+1
    
    return corrmat, corrmat_shape 


def retina_mix(corrmat, corrmat_shape, mix_ratio):
    """
    Mixes the passed correlation matrix in a manner, that the results of regular cross correlation appear in
    arbitrary places.
    
    Args:
    corrmat (numpy.ndarray): Correaltionmatrix, usually by corrmatrix() generated.
    corrmat_shape (list(int)): the shape of the correlationmatrix.
    mix_ratio (float): Ratio for tuning the occurence of reassigning the correlation results.
    
    Returns:
    corrmat (numpy.ndarray): restructured correlation matrix with the same dimensions as before
    corrmat_shape (list(int)): The shape of the 4D correlation matrix. 
    """
    
    Y,X,_y,_x = corrmat_shape
    
    for _ in range(int(mix_ratio*Y*X)):
        rdm_y = np.random.randint(low=0, high=Y, size=[2])
        rdm_x = np.random.randint(low=0, high=X, size=[2])
        buffer = corrmat[rdm_y[0]][rdm_x[0]]
        corrmat[rdm_y[0]][rdm_x[0]] = corrmat[rdm_y[1]][rdm_x[1]]
        corrmat[rdm_y[1]][rdm_x[1]] = buffer
        
    return corrmat, corrmat_shape


def kernel_mix(corrmat, corrmat_shape, kernel_size, mix_ratio):
    """
    Mixes the passed correlation matrix in a manner, that in some cross-correlation operations the kernelshape is varied. 
    
    Args:
    corrmat (numpy.ndarray): Correaltionmatrix, usually by corrmatrix() generated.
    corrmat_shape (list(int)): the shape of the correlationmatrix.
    kernel_size (int): side length of the kernel in pixel.
    mix_ratio (float): Ratio for tuning the occurence of reassigning the correlation results.
    
    Returns:
    corrmat (numpy.ndarray): restructured correlation matrix with the same dimensions as before
    corrmat_shape (list(int)): The shape of the 4D correlation matrix. 
    """
    for _ in range(int(mix_ratio*corrmat_shape[0]*corrmat_shape[1])):
        rdm_frame = np.random.randint(low=0, high=corrmat_shape[0], size=[2])
        rdm_pixel = np.random.randint(low=0, high=corrmat_shape[2], size=[2])
        rdm_kernel = np.random.randint(low=0, high=(kernel_size**2))
        y = rdm_kernel//kernel_size
        x = rdm_kernel%kernel_size
        buffer = corrmat[rdm_frame[0]][rdm_frame[1]][rdm_pixel[0]][rdm_pixel[1]]
        corrmat[rdm_frame[0]][rdm_frame[1]][rdm_pixel[0]][rdm_pixel[1]] = corrmat[rdm_frame[0]][rdm_frame[1]][rdm_frame[0]+y][rdm_frame[1]+x]
        corrmat[rdm_frame[0]][rdm_frame[1]][rdm_frame[0]+y][rdm_frame[1]+x] = buffer
        
    return corrmat, corrmat_shape

@njit
def cross_correlation(input, corrmat, corrmat_shape, kernel, kernel_size):
    """
    performs cross_correlation using an imput and the correlationmatrix with the corresponding weights.
    
    Args:
    input (np.ndarray): 2D input array with Pixelvalues.
    corrmat (np.ndarray): the 4D correlationmatrix as "Linear equation system".
    corrmat_shape (list[int]): The shape of the 4D correlation matrix. 
    kernel (np.ndarray): 2D array with the kernelvalues (weights) in it.
    kernel_size (int): side length of the kernel in pixel.
    
    Returns:
    output (np.ndarray): 2D array as result of applied Cross-Correlation
    """
    kernel = kernel.reshape(kernel_size**2)
    Y, X, y, x = corrmat_shape
    output = np.zeros((Y,X))
    # Compute the cross-correlation for each pixel
    for Y_ in range(Y):
        for X_ in range(X):
            for y_ in range(y):
                for x_ in range(x):
                    val = corrmat[Y_,X_,y_,x_]
                    if(val!= 0):
                        output[Y_, X_] += kernel[val-1]*input[y_][x_]
                        
    return output

@njit
def k_grad_operation(input, output_gradient, corrmat, corrmat_shape, kernel_size):
    """
    Function for calculating the kernelgradient during Backwards process.
    Args:
    input (np.ndarray): 2D input array with Pixelvalues.
    output_gradient (np.ndarray): 2D shaped gradient, returned by previous Layer during backpropagation. 
        Dimensions are equal to the outputdimensions of the forewardpass.
    corrmat (np.ndarray): the 4D correlationmatrix as "Linear equation system".
    corrmat_shape (list[int]): The shape of the 4D correlation matrix. 
    kernel_size (int): side length of the kernel in pixel.
    
    Returns:
    kernel_gradient (np.ndarray): Returns gradient with Dimensions of the kernel.
    """
    Y, X, y, x = corrmat_shape
    kernel_gradient  = np.zeros(kernel_size**2)
    # Compute the gradient of the kernel
    for Y_ in range(Y):
        for X_ in range(X):
            for y_ in range(y):
                for x_ in range(x):
                    val = corrmat[Y_, X_, y_, x_]
                    if(val!=0):
                        kernel_gradient[val-1] += output_gradient[Y_, X_] * input[y_, x_]
    
    kernel_gradient = kernel_gradient.reshape((kernel_size, kernel_size))
    
    return kernel_gradient

# Numba JIT compiled function for computing the gradient of the input
@njit
def x_grad_operation(output_gradient, corrmat, corrmat_shape, kernel):
    """
    output_gradient (np.ndarray): 2D shaped gradient, returned by previous Layer during backpropagation. 
        Dimensions are equal to the outputdimensions of the forewardpass.
    corrmat (np.ndarray): the 4D correlationmatrix as "Linear equation system".
    corrmat_shape (list[int]): The shape of the 4D correlation matrix. 
    kernel (np.ndarray): 2D array with the kernelvalues (weights) in it.
    """
    Y,X,y,x = corrmat_shape
    input_grad = np.zeros((y,x))
    kernel = kernel.reshape(kernel.size)
    
    # Compute the gradient of the input
    for Y_ in range(Y):
        for X_ in range(X):
            for y_ in range(y):
                for x_ in range(x):
                    if(corrmat[Y_][X_][y_][x_]!=0):
                        input_grad[y_][x_] += kernel[corrmat[Y_][X_][y_][x_]-1]*output_gradient[Y_][X_]
                                          
    return input_grad

# Convolutional layer class that inherits from the Layer class
class Convolutional(Layer):
    """
    Convolutional Layer to perform cross-correlation on given input and provide backward pass functionality.
    """
    def __init__(self, input_shape, kernel_size, depth, type = "regular", mix_factor=0):
        """
        Initialize the convolutional layer.

        Args:
            input_shape (tuple): Shape of the input as (input_depth, input_height, input_width).
            kernel_size (int): Size of the convolution kernel.
            depth (int): Number of output feature maps.
            type (str, optional): Type of mixing to apply to the correlation matrix. Options are "regular", "kernelMix", "retinaMix". Default is "regular".
            mix_factor (float, optional): Ratio for mixing elements in the correlation matrix. Default is 0.
        """
        # Initialize the convolutional layer with given parameters
        input_depth, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)   
        # Initialize the correlation matrix based on the type of mixing
        match type:
            case "regular":
                self.corr_matrix, self.corr_matrix_shape = corrmatrix(input_height, input_width, kernel_size)
            case "kernelMix":
                self.corr_matrix, self.corr_matrix_shape = kernel_mix(*corrmatrix(input_height, input_width, kernel_size), self.kernel_size, mix_factor)
            case "retinaMix":
                self.corr_matrix, self.corr_matrix_shape = retina_mix(*corrmatrix(input_height, input_width, kernel_size), mix_factor)
                
    def forward(self, input):
        """
        Perform the forward pass of the convolutional layer.

        Args:
            input (numpy.ndarray): Input data of shape (input_depth, input_height, input_width).

        Returns:
            numpy.ndarray: Output data after applying convolution, of shape (depth, output_height, output_width).
        """
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += cross_correlation(self.input[j], self.corr_matrix, self.corr_matrix_shape, self.kernels[i,j], self.kernel_size)
        return self.output
    
    def backward(self, output_gradient):
        """
        Perform the backward pass of the convolutional layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss with respect to the output, of shape (depth, output_height, output_width).

        Returns:
            tuple: Gradients with respect to the input and kernels.
                - input_gradient (numpy.ndarray): Gradient with respect to the input, of shape (input_depth, input_height, input_width).
                - kernels_gradient (numpy.ndarray): Gradient with respect to the kernels, of shape (depth, input_depth, kernel_size, kernel_size).
        """
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = k_grad_operation(self.input[j], output_gradient[i], self.corr_matrix, self.corr_matrix_shape, self.kernel_size)
                input_gradient[j] += x_grad_operation(output_gradient[i], self.corr_matrix, self.corr_matrix_shape, self.kernels[i, j])
        
        return input_gradient, kernels_gradient
    
    def learning(self, weight_grad, bias_grad):
        """
        Update the weights and biases of the convolutional layer.

        Args:
            weight_grad (numpy.ndarray): Gradient of the loss with respect to the weights.
            bias_grad (numpy.ndarray): Gradient of the loss with respect to the biases.
        """
        self.kernels -= weight_grad
        self.biases -= bias_grad
