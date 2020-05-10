from nose.tools import *
import NeuralNetwork as NN
import numpy as np
import random

def neuron_test():

    """
    Create neurons
    """

    #test function
    sigmoid = NN.model.sigmoid
    real = 0.6224593312018546
    eq_(real, sigmoid(0.5))

    #create a neuron
    neuron = NN.model.Neuron(sigmoid)
    real_and_bias   =  0.8175744761936436596072
    eq_(neuron.a(0.5), real_and_bias) #real + bias

def layer_test():

    """
    Create and test layers
    """
    #test input layer
    #create dataset object
    filename = "dataset/iris.data"
    dataset = NN.Dataset(filename)

    #exemple to input
    input = dataset.data[0,:-1].astype(float)

    #****************************************
    #*************Test layers ***************
    #****************************************

    #create model
    model = NN.model

    #*************input layer****************
    nb_arguments = len(input)
    l_input = model.Layer("input", nb_arguments)

    #weights of the input layer are all 1
    eq_(model.layers[0].weights, [1.,1.,1.,1.])

    #output of the input layer should be equal to the input values
    arrays_equal = np.array_equal(l_input.get_output(input), input)
    eq_(arrays_equal, True)

    #**************hidden layer****************
    #create weights matrix of size  (5,2) ==> (4 args of input + 1 bias, 2 neurons )
    np.random.seed(1)
    weights = np.round(np.random.rand(5,2),3)

    #create hidden layer with 2 neurons
    l_hidden = model.Layer("hidden", 2, model.sigmoid)
    eq_(model.layers[1].weights.all(), weights.all())

    #dot product of weights (including bias) and input (expected value )
    #dot products calculated by hand
    output = model.sigmoid(np.array([2.767, 5.466]).flatten())

    #test output
    arrays_equal = np.array_equal(model.layers[-1].get_output(input), output)
    eq_(arrays_equal, True)

    #**************output layer****************
    # output of previous layer becomes input of output layer
    input = model.layers[-1].get_output(input)

    #get number of classes
    _ , counts = np.unique(dataset.data[:,-1], return_counts = True)
    NbClasses = len(counts)

    #create weights that will be created in the layer
    np.random.seed(1)
    weights = np.round(np.random.rand(model.layers[-1].NbNeurons + 1, NbClasses),3)

    #expected output final layer
    o_prev = np.array([0.941, 0.996, 1])
    sum = np.round(np.dot(o_prev,weights),3)
    expected_o_final = model.sigmoid(sum)

    #create output layer
    NbNeurons = NbClasses
    l_output = model.Layer("output",  NbNeurons, model.sigmoid)

    #get output of final layer
    real_o_final = l_output.get_output(input)

    #test output
    arrays_equal = np.array_equal(real_o_final, expected_o_final)
    eq_(arrays_equal, True)
