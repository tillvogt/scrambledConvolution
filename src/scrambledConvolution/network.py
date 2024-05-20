import timeit
import numpy as np

from .layers import Convolutional

import timeit
import numpy as np
from .layers import Convolutional

def timer_decorator(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (callable): The function to measure.

    Returns:
        callable: The wrapped function with timing.
    """
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"Time taken: {end_time - start_time}")
        return result
    return wrapper

def predict(network, input):
    """
    Perform a forward pass through the network.

    Args:
        network (list): List of layers in the network.
        input (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Output after the forward pass.
    """
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def data_batcher(x_train, y_train, batch_size):
    """
    Batch the training data.

    Args:
        x_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training labels.
        batch_size (int): Size of each batch.

    Returns:
        tuple: Batched training input and labels.
    """
    length = x_train.shape[0] // batch_size
    batched_x = np.empty(length, dtype=object)
    batched_y = np.empty(length, dtype=object)
    
    for batch in range(length):
        low = batch * batch_size
        high = low + batch_size
        batched_x[batch] = x_train[low:high]
        batched_y[batch] = y_train[low:high]
    
    return batched_x, batched_y

def batch_train(network, batched_data, batch_size, loss, loss_prime, learning_rate, friction):
    """
    Train the network in batches.

    Args:
        network (list): List of layers in the network.
        batched_data (zip): Batched training data as a zip of inputs and labels.
        batch_size (int): Size of each batch.
        loss (callable): Loss function.
        loss_prime (callable): Derivative of the loss function.
        learning_rate (float): Learning rate for gradient descent.
        friction (float): Momentum factor for gradient updates.

    Returns:
        float: Total error for the training data.
    """
    error = 0
    network_size = len(network)
    
    for batch_x, batch_y in batched_data:
        input_gradient = None
        weight_gradient = None
        
        batch_input_grads = np.empty((batch_size, network_size), dtype=object)
        batch_weight_grads = np.empty((batch_size, network_size), dtype=object)
        
        for n in range(batch_size):
            # Forward pass
            output = predict(network, batch_x[n])

            # Compute error
            error += loss(batch_y[n], output)
            
            # Backward pass
            grad = loss_prime(batch_y[n], output)
            batch_input_grads[n, network_size-1] = grad
            for i, layer in enumerate(reversed(network)):
                batch_input_grads[n, network_size-2-i], batch_weight_grads[n, network_size-1-i] = layer.backward(batch_input_grads[n, network_size-1-i])
                    
        mean_input_gradient = np.sum(batch_input_grads, axis=0) / batch_size
        mean_weight_gradient = np.sum(batch_weight_grads, axis=0) / batch_size
        
        if input_gradient is None:
            input_gradient = mean_input_gradient
            weight_gradient = mean_weight_gradient
        else:
            input_gradient = friction * input_gradient + mean_input_gradient
            weight_gradient = friction * weight_gradient + mean_weight_gradient
        
        for i, layer in enumerate(reversed(network)):
            layer.learning(weight_gradient[network_size-1-i] * learning_rate, input_gradient[network_size-1-i] * learning_rate)    
        
    return error

# Main training function with timing decorator
@timer_decorator
def train(mixing_factor, network, loss, loss_prime, x_train, y_train, x_test, y_test, 
          epochs, learning_rate=0.1, batch_size=4, friction=0.9,
          verbose=True, weight_saving=True):
    """
    Train a neural network with the given parameters and data.

    Args:
        mixing_factor (float): Factor for mixing data.
        network (list): List of layers in the neural network.
        loss (callable): Loss function.
        loss_prime (callable): Derivative of the loss function.
        x_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test input data.
        y_test (numpy.ndarray): Test labels.
        epochs (int): Number of training epochs.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.1.
        batch_size (int, optional): Size of each batch. Defaults to 4.
        friction (float, optional): Momentum factor for gradient updates. Defaults to 0.9.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
        weight_saving (bool, optional): Whether to save weights of convolutional layers. Defaults to True.

    Returns:
        tuple: Error statistics and accuracy statistics. If `weight_saving` is True, also returns the weights of convolutional layers.
    """
    
    err_stats, acc_stats = [], []
    batched_x, batched_y = data_batcher(x_train, y_train, batch_size)
    
    for e in range(epochs):
        train_data = zip(batched_x, batched_y)
        test_data = zip(x_test, y_test)
        
        # Train
        error = 0
        error += batch_train(network, train_data, batch_size, loss, loss_prime, learning_rate, friction)                   
        error /= len(x_train)
        
        # Test
        test_error = 0
        true_results = 0
        for x, y in test_data:
            output = predict(network, x)
            test_error += loss(y, output)
            
            output_index = np.argmax(output)
            test_index = np.argmax(y)
            if output_index == test_index:
                true_results += 1

        accuracy = (true_results / y_test.shape[0]) * 100
        test_error /= len(x_test)
        
        if verbose:
            print(f"{mixing_factor:.2f}, {e + 1}/{epochs}, train_error={error:.4f}, test_error={test_error:.4f}, accuracy={accuracy:2n}%")
                
        err_stats.append(error)
        acc_stats.append(test_error)
            
    if weight_saving == False:    
        return err_stats, acc_stats
    
    weights = []
    for x in network:
        if isinstance(x, Convolutional) and weight_saving:
            weights.append([x.kernels, x.biases])            

    return err_stats, acc_stats, weights