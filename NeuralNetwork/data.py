import numpy as np


class Dataset(object):

    def __init__(self,filename , _dtype = "S12", _delimiter = ','):

        #class is always in the last column
        self.classPosition = -1

        #Try to read filename onto a numpy array otherwise show error
        try:
            self.data = np.loadtxt(filename, dtype = _dtype, delimiter=_delimiter)
            self.data[:,:-1] = self.data[:,:-1].astype(float)
        except ValueError:
            print("Error loading dataset.")

        #Get dataset information (number attributes, size, etc)
        self.dataset_info()

    def dataset_info(self):
        """Function that defines all the info of the dataset so it is ready when needed"""

        #identify classes
        self.classes = np.unique(self.data[:,self.classPosition])
        self.NbClasses = len(self.classes)
