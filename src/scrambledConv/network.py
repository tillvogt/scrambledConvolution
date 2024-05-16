import timeit
import numpy as np

from .layers import Convolutional

# Decorator to measure the execution time of a function
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"Time taken: {end_time - start_time}")
        return result
    return wrapper

# Function to perform a forward pass through the network
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# Function to batch the training data
def data_batcher(x_train, y_train, batch_size):
    length = x_train.shape[0] // batch_size
    batched_x = np.empty(length, dtype=object)
    batched_y = np.empty(length, dtype=object)
    for batch, _ in enumerate(batched_x):
        low = batch * batch_size
        high = low + batch_size
        batched_x[batch] = x_train[low:high]
    for batch, _ in enumerate(batched_y):
        low = batch * batch_size
        high = low + batch_size
        batched_y[batch] = y_train[low:high]
        
    return batched_x, batched_y
    
# Function to train the network in batches
def batch_train(network, batched_data, batch_size, loss, loss_prime, learning_rate, friction):
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
            for _, layer in enumerate(reversed(network)):
                batch_input_grads[n, network_size-2-_], batch_weight_grads[n, network_size-1-_] = layer.backward(batch_input_grads[n, network_size-1-_])
                    
        mean_input_gradient = np.sum(batch_input_grads, axis=0) / batch_size
        mean_weight_gradient = np.sum(batch_weight_grads, axis=0) / batch_size
        
        if input_gradient is None:
            input_gradient = mean_input_gradient
            weight_gradient = mean_weight_gradient
        else:
            input_gradient = friction * input_gradient + mean_input_gradient
            weight_gradient = friction * weight_gradient + mean_weight_gradient
        
        for _, layer in enumerate(reversed(network)):
            layer.learning(weight_gradient[network_size-1-_] * learning_rate, input_gradient[network_size-1-_] * learning_rate)    
        
    return error

# Main training function with timing decorator
@timer_decorator
def train(factor, network, loss, loss_prime, x_train, y_train, x_test, y_test, 
          epochs, learning_rate=0.1, batch_size=4, friction=0.9,
          verbose=True, weight_saving=True):
    
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
            print(f"{factor:.2f}, {e + 1}/{epochs}, train_error={error:.4f}, test_error={test_error:.4f}, accuracy={accuracy:2n}")
                
        err_stats.append(error)
        acc_stats.append(test_error)
            
    if weight_saving == False:    
        return err_stats, acc_stats
    
    weights = []
    for x in network:
        if isinstance(x, Convolutional) and weight_saving:
            weights.append([x.kernels, x.biases])            

    return err_stats, acc_stats, weights
