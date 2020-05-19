import numpy as np
import random
from . import fit_model

# random.seed(1)
# np.random.seed(1)

alpha = 0.0001
beta = 0.999
#initialize model layer array where all the layers will be stored
layers = []


def fit(exemples, train_labels, expected, NbEpochs = 800):
    """function that trains the neural network model and sets its weights
    args
    exemples: instances of a class (attributes)
    train_labels: class of the exemples given
    expected: dictionary of the output expected in the class list
    NbEpochs: number of epochs for backpropagation"""

    #for each epoch
    for i in range(NbEpochs):
        #shuffle list
        randomize = np.arange(len(exemples))
        exemples = exemples[randomize]
        classes = classes[randomize]

        #initialize error
        error_ = 0

        #backpropagate each exemple
        for exemple, class_ in zip(exemples, classes):
            #get output
            output = feedforward(exemple)

            #add error to global error
            error_ += np.sum(error(output, expected[class_]))

            #backpropagate with exepected class
            backpropagation(expected[class_])

        #print 10 epochs and there respetive error
        if i % int(NbEpochs/10) == 0:
            print("epoch number {} and loss {}".format(i,error_/len(exemples)))


class Layer():

    def __init__(self, type, NbNeurons, g = None):
        self.type = type

        #Create neurons and add to list
        self.weights = []
        self.NbNeurons = NbNeurons
        self.g = g #activation function
        self.a = []
        self.sum = []
        #assign weights to layer
        self.assign_weights()

    def assign_weights(self):

        #input layer
        if self.type == "input":
            for _ in range(self.NbNeurons):
                self.weights.append(1)
            layers.append(self)

        #hidden and ouput layer
        else:
            for _ in range(self.NbNeurons):

            #get number of outputs from the previous layer
            NbNeurons_prev = layers[-1].NbNeurons
            #create matrix of weights including the bias
            self.weights = np.random.rand(NbNeurons_prev + 1,self.NbNeurons)
            #for momentum
            self.z = np.zeros(np.shape(self.weights))
            self.delta_W = np.zeros(np.shape(self.weights))

            #add to global model
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

            #get sum to the imput of the neurons
            self.sum = input.dot(self.weights)

            #activation function
            self.a = self.g(self.sum) #activation
            return self.a

def sigmoid(sum):
    return 1/( 1 + np.exp(-sum) )

def der_sigmoid(sum):
    return sigmoid(sum) * (1 - sigmoid(sum))

def input_activation(arg):
    return arg

def feedforward(input):

    """perform feedforward
    args
    input: attributes of a given instance """


    for layer in layers:
        output = layer.get_output(input)
        input = output
    return layers[-1].a

def error(y , y_):
    """compute error for an output and the expected value for such exemple"""
    return (y-y_)**2

def backpropagation(y):
    """backpropagation """

    """carry gradient from last layer """
    #get derivative of the output of previous layer and append 1 for bias
    a = der_sigmoid(layers[-1].sum)
    # the derivative of the error with respect to the output
    dL_dy = -(y - layers[-1].a)
    #multiply to get delta j
    delta_j = (a*dL_dy).flatten() #make 1 dimension

    #multiply by output of previous layer (append 1 for bias), then by alpha and add to matrix of weight
    #first save weights before updating them for next iteration
    upper_layer_weights = layers[-1].weights
    delta_W =  np.outer(np.append(layers[-2].a,1),delta_j) #********checked

    #weights update with momentum
    layers[-1].z = (beta * layers[-1].z) + layers[-1].delta_W
    #add z to weights
    layers[-1].weights = layers[-1].weights - ( alpha * layers[-1].z)
    #save delta W in layer object for next epoch
    layers[-1].delta_W = delta_W

    """get gradient of previous layer with global gradient"""
    delta_i = delta_j

    for i in range(len(layers)-2,0,-1):
        #get delta i of each neuron
        delta_i = (delta_i @ upper_layer_weights[:-1].T) * der_sigmoid(layers[i].sum)
         #save for next delta i
        upper_layer_weights = layers[i].weights

        #compute delta W
        delta_W =  np.outer(np.append(layers[i-1].a,1),delta_i)
        #comput z for momentum
        layers[i].z = (beta * layers[i].z) + layers[i].delta_W
        #add z to weights
        layers[i].weights = layers[i].weights - (alpha * layers[i].z)
        #save delta W in layer for next z in next epoch
        layers[i].delta_W = delta_W
