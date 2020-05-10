from nose.tools import *
import NeuralNetwork as NN
import numpy as np
import random


def feedforward_test():

    """
    Tests feedforward
    """

    #create dataset object
    filename = "dataset/iris.data"
    dataset = NN.Dataset(filename)

    #exemple to input
    input = np.array([0,1,1])
    nb_arguments = len(input)

    #get number of classes
    NbClasses = 3

    #create model
    model = NN.model

    #add input layer
    model.Layer("input", nb_arguments)

    #add hidden layer with 1 neurons
    model.Layer("hidden", 4, model.sigmoid)

    #add hidden layer with 1 neurons
    model.Layer("hidden", 4, model.sigmoid)
    #create output layer
    NbNeurons = NbClasses
    model.Layer("output",  NbNeurons, model.sigmoid)

    #feedforward
    output = model.feedforward(input)

    return input, output, model

def backpropagation_test():
    """Test backpropagation"""

    input, output, model = feedforward_test() #get both from previous function
    y = np.array([0,1,0])
    error = model.error(output, y)
    print("error before backpropagation: ",np.round(error,3))

    for i in range(1000):
        model.backpropagation(y)
        output = model.feedforward(input) #get both from previous function

    error = model.error(output, y)
    print("error after backpropagation", np.round(error,3))
