from nose.tools import *
import NeuralNetwork as NN
import numpy as np


def dataset_test():

    """
    Test function that loads dataset
    """
    #create dataset object
    filename = "dataset/iris.data"
    dataset = NN.Dataset(filename)

    #148 exemples for training
    eq_((148,5),np.shape(dataset.data))

    
