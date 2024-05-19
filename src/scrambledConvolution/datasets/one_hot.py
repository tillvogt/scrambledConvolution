import numpy as np

def to_categorical(y_array):
    one_hot = np.zeros((len(y_array), 10, 1))
    for _, i in enumerate(y_array):
        one_hot[_, i, 0] = 1
        
    return one_hot      