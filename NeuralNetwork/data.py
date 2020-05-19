import numpy as np


class Dataset(object):

    def __init__(self,filename , _dtype = "S12", _delimiter = ','):

        #class is always in the last column
        self.classPosition = -1

        #Try to read filename onto a numpy array otherwise show error
        try:
            self.data = np.loadtxt(filename, dtype = _dtype, delimiter=_delimiter)
            np.random.shuffle(self.data) #shuffle
            self.exemples = np.float_(self.data[:,:-1])
            self.labels = self.data[:,-1]

        except ValueError:
            print("Error loading dataset.")

        #Get dataset information (number attributes, size, etc)
        self.dataset_info()
        self.create_expected_output()

    def dataset_info(self):
        """Function that defines all the info of the dataset so it is ready when needed"""

        #identify classes
        self.classes = np.unique(self.data[:,self.classPosition])
        self.NbClasses = len(self.classes)

        #arguments
        self.NbArguments  = len(self.exemples[0])

        #total number of exemples in dataset
        self.total_ex = len(self.data)


    def create_expected_output(self):
        """creates an dictionary with the expected outputs of the dataset"""

        # create empty dictionary
        self.expected = {}

        #for each class in the class repertoire
        for class_, i in zip(np.unique(self.data[:,-1]),range(self.NbClasses)):
            #create output with the number of classes all set to zero
            output = np.zeros(self.NbClasses).astype(int)
            #The class will have an output neuron set to one according to their order in the class list
            output[i] = 1
            #assign output to the dictionary 
            self.expected[class_] = output
