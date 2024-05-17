import numpy as np
import os


def load_data(*args, **kwargs):
    dir_path = os.path.dirname(__file__)
    with np.load(dir_path+"/mnist.npz") as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    