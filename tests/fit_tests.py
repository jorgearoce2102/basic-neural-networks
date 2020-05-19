from nose.tools import *
import NeuralNetwork as NN
import numpy as np
import random

def with_sgd_test():
    """Stocastic gradient descent backpropagation test"""

    #create dataset object
    filename = "dataset/iris.data"
    dataset = NN.Dataset(filename)

    #training and testing datasets
    train_ratio = 0.75
    train = dataset.exemples[:int(train_ratio*dataset.total_ex),:]
    train_labels = dataset.labels[:int(train_ratio*dataset.total_ex)]

    test = dataset.exemples[int(train_ratio*dataset.total_ex):,:]
    test_labels = dataset.labels[int(train_ratio*dataset.total_ex):]

    #get number of classes
    NbClasses = dataset.NbClasses
    expected = dataset.expected
    NbArguments = dataset.NbArguments

    #create model
    model = NN.model

    #add input layer
    model.Layer("input", NbArguments)

    #add hidden layer with 4 neurons
    model.Layer("hidden", 6, model.sigmoid)

    #add hidden layer with 4 neurons
    model.Layer("hidden", 4, model.sigmoid)


    #create output layer
    NbNeurons = NbClasses
    model.Layer("output",  NbNeurons, model.sigmoid)

    model.fit(train, train_labels, expected)

    return model, test, test_labels, expected

def confusion_matrix_test():
    "get the confusion matrix of the model after training"

    #get model and test info
    model, test, test_labels, expected = with_sgd_test()

    #initialize confution matrix
    classes = len(np.unique(test_labels))
    confusion_matrix = np.zeros((classes,classes), int)

    error = 0
    for exemple, label in zip(test, test_labels):
        #get output from feedforward, flatten it to round it, then make it integer integer
        output = (np.round(model.feedforward(exemple).flatten())).astype(int)

        #get error
        error += np.sum(output - expected[label])


        try:
            confusion_matrix[np.where(output==1)[0][0]][np.where(expected[label]==1)[0][0]] += 1
        except:
            print("undefined class")

    print(confusion_matrix)
