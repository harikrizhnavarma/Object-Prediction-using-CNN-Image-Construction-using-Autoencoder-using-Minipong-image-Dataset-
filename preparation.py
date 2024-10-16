import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder

class DataSet:
    def __init__(self):
        self.trainPix = np.genfromtxt("trainingpix.csv", delimiter = ",", skip_header = 0)
        self.testPix = np.genfromtxt("testingpix.csv", delimiter = ",", skip_header = 0)
        self.trainLabels = np.genfromtxt("traininglabels.csv", delimiter = ",", skip_header = 0)
        self.testLabels = np.genfromtxt("testinglabels.csv", delimiter = ",", skip_header = 0)

        self.AllPix = np.genfromtxt("allpix.csv", delimiter = ",", skip_header = 0)
    
    def dataPreProcess(self):

        self.trainPix = self.trainPix.reshape(-1,1,15,15)
        self.testPix = self.testPix.reshape(-1,1,15,15)

        encode = OneHotEncoder(sparse_output = False)
        
        # encode x label data
        self.xTrainLabels = encode.fit_transform(self.trainLabels[:, 0].reshape(-1, 1))
        self.xTestLabels = encode.fit_transform(self.testLabels[:, 0].reshape(-1, 1))

        # encode y label data
        self.yTrainLabels = encode.fit_transform(self.trainLabels[:, 1].reshape(-1, 1))
        self.yTestLabels = encode.fit_transform(self.testLabels[:, 1].reshape(-1, 1))

        # encode z label data
        self.zTrainLabels = encode.fit_transform(self.trainLabels[:, 2].reshape(-1, 1))
        self.zTestLabels = encode.fit_transform(self.testLabels[:, 2].reshape(-1, 1))

        # convert to pytorch tensors
        self.trainPix = torch.tensor(self.trainPix, dtype = torch.float32)
        self.testPix = torch.tensor(self.testPix, dtype = torch.float32)

        # for x label data
        self.xTrainLabels = torch.tensor(self.xTrainLabels, dtype = torch.float32)
        self.xTestLabels = torch.tensor(self.xTestLabels, dtype = torch.float32)

        # for y label data
        self.yTrainLabels = torch.tensor(self.yTrainLabels, dtype = torch.float32)
        self.yTestLabels = torch.tensor(self.yTestLabels, dtype = torch.float32)

        # for z label data
        self.zTrainLabels = torch.tensor(self.zTrainLabels, dtype = torch.float32)
        self.zTestLabels = torch.tensor(self.zTestLabels, dtype = torch.float32)

        # combine the data for x labels
        self.xTrainSet = TensorDataset(self.trainPix, self.xTrainLabels)
        self.xTestSet = TensorDataset(self.testPix, self.xTestLabels)
        
        # combine the data for y labels
        self.yTrainSet = TensorDataset(self.trainPix, self.yTrainLabels)
        self.yTestSet = TensorDataset(self.testPix, self.yTestLabels)

        # combine the data for x labels
        self.zTrainSet = TensorDataset(self.trainPix, self.zTrainLabels)
        self.zTestSet = TensorDataset(self.testPix, self.zTestLabels)

        #lad the data for x labels
        self.xTrainLoader = DataLoader(self.xTrainSet, batch_size = 32, shuffle = True)
        self.xTestLoader = DataLoader(self.xTestSet, batch_size = 32, shuffle = True)

        #lad the data for x labels
        self.yTrainLoader = DataLoader(self.yTrainSet, batch_size = 32, shuffle = True)
        self.yTestLoader = DataLoader(self.yTestSet, batch_size = 32, shuffle = True)

        #lad the data for x labels
        self.zTrainLoader = DataLoader(self.zTrainSet, batch_size = 32, shuffle = True)
        self.zTestLoader = DataLoader(self.zTestSet, batch_size = 32, shuffle = True)

        return self.xTrainLoader, self.xTestLoader, self.yTrainLoader, self.yTestLoader, self.zTrainLoader, self.zTestLoader
    
    def allDataPreProcess(self):

        self.AllPix = self.AllPix.reshape(-1,1,15,15)
        self.AllPix = torch.tensor(self.AllPix, dtype = torch.float32)
        self.AllPix = TensorDataset(self.AllPix)
        self.AllDataLoader = DataLoader(self.AllPix, batch_size = 32, shuffle = True)

        return self.AllDataLoader

