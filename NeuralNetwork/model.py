import numpy as np
import random

random.seed(1)
np.random.seed(1)
alpha = 0.2
#initialize model layer array where all the layers will be stored
layers = []

class Neuron():
    def __init__(self, g):
        """Create the neuron unit.
        args
        g: activation function
        """
        #activation function (sigmoid)
        self.g = g

    def a(self, sum):
        """activate neuron. args
        sum: sum of the previous layer"""

        return self.g(sum)

class Layer():

    def __init__(self, type, NbNeurons, g = None):
        self.type = type

        #Create neurons and add to list
        self.neurons = []
        self.weights = []
        self.NbNeurons = NbNeurons
        self.g = g #activation function
        self.a = []
        self.sum = []
        #assign weights tp
        self.assign_weights()

    def assign_weights(self):

        #input layer
        if self.type == "input":
            for _ in range(self.NbNeurons):
                neuron = Neuron(input_activation)
                self.neurons.append(neuron)
                self.weights.append(1)
            layers.append(self)

        #hidden and ouput layer
        else:
            for _ in range(self.NbNeurons):
                neuron = Neuron(self.g) #sigmoid activation
                self.neurons.append(neuron)

            #get number of outputs from the previous layer
            NbNeurons_prev = layers[-1].NbNeurons
            np.random.seed(1)                                                    #TODO erase seed
            #create matrix of weights including the bias
            self.weights = np.random.rand(NbNeurons_prev + 1,self.NbNeurons)
            self.weights = np.round(self.weights,3) #round to the 3rd digit
            layers.append(self)

    def get_output(self, input):

        #input layer  output equal to input
        if self.type == "input":
            self.a = input
            return self.a

        #hidden and output layer
        else:
            #append for bias and reshape
            input = np.append(input, 1).reshape((1,len(self.weights)))
            self.sum = np.round(np.dot(input,self.weights).flatten(), 3)
            self.a = self.g(self.sum) #activation
            return self.a

def sigmoid(sum):
    return np.round(1/( 1 + np.exp(-sum) ),3)

def der_sigmoid(sum):
    return sigmoid(sum) * (1 - sigmoid(sum))

def input_activation(arg):
    return arg

def feedforward(input):

    """perform feedforward"""
    output = np.empty(0)
    for layer in layers:
        output = layer.get_output(input)
        input = output
    return output

def error(y , y_):
    return (y-y_)**2

def backpropagation(y):
    """backpropagation """

    """carry gradient from last layer """
    #get derivative of the output of previous layer and append 1 for bias
    a = np.append(der_sigmoid(layers[-2].a), 1)

    #multiply the derivative of the error with respect to the output
    delta_j = np.outer((y - layers[-1].a),a).sum(axis=1)

    #multiply by output of previous layer (append 1 for bias), then by alpha and add to matrix of weights
    layers[-1].weights = layers[-1].weights.__add__(alpha * np.outer(np.append(layers[-2].a,1),delta_j))

    """get gradient of previous layer with global gradient"""
    delta_i = delta_j

    for i in range(len(layers)-2,1):
        delta_i = delta_i @ der_sigmoid(layers[i-1].a)
        layers[1].weights += alpha * delta_i * layers[i].a
