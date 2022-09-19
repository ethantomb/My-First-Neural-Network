
import time
import math
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""
Hidden Layer Object
Holds the W,B,A,Z values for each layer
W - [n Prev Layer Nodes X n current layer nodes] Matrix - Weights for the current layer
B - [n current Layer Nodes X 1] Matrix - Bias values for each node
Z - [N current Nodes x Input Size]  Matrix - Raw Result of WX + B for each node
A - |Z| Matrix - Values of Z passed through the activation function
"""
class Layer:
    """
    Layer Constructor
    PrevLayerSize, ThisLayerSize - Number of nodes in the preceding layer, number of nodes in this layer
    """
    def __init__(self,PrevLayerSize,ThisLayerSize):
        self.W = np.random.rand(ThisLayerSize,PrevLayerSize)-0.5
        self.B = np.random.rand(ThisLayerSize,1)-0.5
        self.A = None
        self.Z= None
    """
    Updates the layer's weights and biases as part of the backpropogation process.
    dW - Deriv. of the weight Values
    dB - Deriv. of the bias values
    learnRate - scalar multiple of dW and dB by which to subtract from W and B
    """
    def update(self,dW,dB,learnRate):
        self.W= self.W - learnRate*dW
        self.B= self.B - learnRate* dB

"""
An N-Dimmensional Feed Forward Neural Network.

"""
class NeuralNetwork:
    
    
    def __init__(self,TrainDat=None,Labels=None,Layers=None,learnRate=None,trainingSplit=0.8):
        """
        NeuralNetwork Constructor
        [Numpy ndARRAY] TrainDat - The input data to train the network on. Expects an array of [N input Nodes X N Input datapoints]
        [Numpy ndARRAY] Labels - The labels for the input data. Expects an array of [1x N datapoints]
        [Int[]] Layers - The Number of nodes In each layer. Expects [Hidden Layer(hl) 1 Nodes,...,hl n Nodes,Output Layer Nodes]
        [Float] learnRate - The Learning Rate for the network
        [Float] trainingSplit - percentage of data to use for validation vs train
        """
        scaler = preprocessing.MinMaxScaler()
        self.Input = scaler.fit_transform(TrainDat)
        self.Labels = Labels
        nTrain = int(self.Input.shape[1] * trainingSplit)
        
        self.HiddenLayers = []
        self.alpha = learnRate
        #Initialize each hidden layer with weight bias values.
        self.HiddenLayers.append(Layer(self.Input.shape[0],Layers[0]))
        for i in range(1,len(Layers)):
            self.HiddenLayers.append(Layer(Layers[i-1],Layers[i]))
        self.Ans = np.zeros((self.Labels.size,Layers[len(Layers)-1]))
        self.Ans[np.arange(self.Labels.size), self.Labels] = 1
        self.Ans=self.Ans.T
        self.outI = len(self.HiddenLayers)-1
        self.ValidationData = self.Input[:,nTrain:]
        self.ValidationLabels = self.Labels[:,nTrain:]
        self.ValidationAns = self.Ans[:,nTrain:]
        self.Input = self.Input[:,:nTrain]
        self.Labels = self.Labels[:,:nTrain]
        self.Ans = self.Ans[:,:nTrain]
    def load(self,path,nLayers):
        """
        Loads a pretrained network configuration using .csv weights and biases files.
        path - The path to the weights and biases. eg: `output/2Hidden1Out`. Do not include the Wn.csv or Bn.csv ending
        nLayers - The number of weights and biases files to look for. You could break this, but why would you want to, huh?
        """
        self.HiddenLayers = []
        for i in range(0,nLayers):
            self.HiddenLayers.append(Layer(1,1))
            self.HiddenLayers[i].W = np.array(pd.read_csv(path+"W"+str(i+1)+".csv",header=None))
            self.HiddenLayers[i].B = np.array(pd.read_csv(path+"B"+str(i+1)+".csv",header=None))
        self.outI = len(self.HiddenLayers)-1
        return
    
    def forwardProp(self):
        """
        Linearly transforms a set of inputs X_0 through each layer X_n of weights W_n and biases B_n to 
        obtain the output predictions. 
        """
        assert self.Input is not None
        #First hidden layer Node Values
        self.HiddenLayers[0].Z = self.HiddenLayers[0].W.dot(self.Input)+self.HiddenLayers[0].B
        #First hidden layer activation. Sets all negative Z values to zero.
        self.HiddenLayers[0].A = self.relU(self.HiddenLayers[0].Z)
        for i in range(1,len(self.HiddenLayers)-1):
            #Z_n = W_n*X_n-1+B_n
            self.HiddenLayers[i].Z = self.HiddenLayers[i].W.dot(self.HiddenLayers[i-1].A)+self.HiddenLayers[i].B
            #A_n = relU(Z_n)
            self.HiddenLayers[i].A = self.relU(self.HiddenLayers[i].Z)
        #Do the final layer last because different activation function
        #Z_out = W_n*X_n-1+B_n
        self.HiddenLayers[self.outI].Z = self.HiddenLayers[self.outI].W.dot(self.HiddenLayers[self.outI-1].A)+self.HiddenLayers[self.outI].B
        #A_out = W_n*X_n-1+B_n
        self.HiddenLayers[self.outI].A = self.softmax(self.HiddenLayers[self.outI].Z)
    
    def relU(self,Z):
        """
        RelU Activation function
        Calculates Max(a,0) for all a in Z
        """
        return np.maximum(Z,0)
    
    def sig(self,Z):
        """
        Unused Sigmoid Activation Function
        """    
        A = 1/(1+np.exp(-Z))
        return A
    
    def softmax(self,Z):
        """
        Softmax Function
        Calculates the probability of each a in Z by the equation pi = e^xi/sum_1-n(e^xi).
        Used here to calculate the highest weighted output node
        """
        M = np.exp(Z)/sum(np.exp(Z))
        return M
    
    def get_predictions(self):
        """
        Returns the highest probability output values for each datapoint.
        """
        return np.argmax(self.HiddenLayers[self.outI].A, 0)
    
    def backProp(self):
        """
        Back Propogation for the network
        Calculates the error cost = Predictions-Correct Answers and changes the weights and biases to reduce the cost
        """
        assert self.Input is not None
        assert self.Ans is not None
        
        batchSize = self.Input.shape[1]
        dWs = []
        dBs = []
        #Initial Cost: Predictions-Correct Answers
        dZ = self.HiddenLayers[self.outI].A-self.Ans
        #Change in output layer weights = 1/|datapoints| * cost*A_out-1
        dW = (1.0/batchSize)*dZ.dot(self.HiddenLayers[self.outI-1].A.T)
        #Change in bias = average cost
        dB = (1.0/batchSize)*np.sum(dZ)
        #push change in weights, change in bias to lists, update layer weights at the end
        dBs.insert(0,np.copy(dB))
        dWs.insert(0,np.copy(dW))
        #Calculate changes to weights and biases for each layer
        for i in range(self.outI-1,0,-1):
            #Layer n Cost = Next Layer Weights*Layer N+1 cost * Active Z nodes 
            dZNext = self.HiddenLayers[i+1].W.T.dot(dZ)*self.Activation(self.HiddenLayers[i].Z)
            dW = (1.0/batchSize)*dZNext.dot(self.HiddenLayers[i-1].A.T)
            dB = (1.0/batchSize)*np.sum(dZNext)
            dBs.insert(0,np.copy(dB))
            dWs.insert(0,np.copy(dW))
            
            dZ = np.copy(dZNext)

        dZNext = self.HiddenLayers[1].W.T.dot(dZ)*self.Activation(self.HiddenLayers[0].Z)
        
        dW = (1.0/batchSize)*dZNext.dot(self.Input.T)
        dB = (1.0/batchSize)*np.sum(dZNext)
        
        dBs.insert(0,np.copy(dB))
        dWs.insert(0,np.copy(dW))
        
        #Update the Weights and Biases for each layer
        for i in range(0,len(self.HiddenLayers)):
            self.HiddenLayers[i].update(dWs[i],dBs[i],self.alpha)
    
    def Activation(self,Z):
        """
        Activation Function

        """
        return Z>0
    
    def accuracy(self):
        """
        Gets the acuracy of the network
        acuracy = num correct predictions/total predictions
        """
        predict = self.get_predictions()
        return np.sum(predict==self.Labels)/self.Labels.size
   
    def saveConfiguration(self,filename):
        """
        Saves the current configuration to output/filenameWn.csv, output/filenameBn.csv
        filename - The name of the output file
        """
        for i,layer in enumerate(self.HiddenLayers):
            np.savetxt("./output/"+filename+"W"+str(i+1)+".csv",layer.W,delimiter=",")
            np.savetxt("./output/"+filename+"B"+str(i+1)+".csv",layer.B,delimiter=",")
        
        print("Files saved to ./output/"+filename+"_.csv")
    def classify(self,inputParameter):
        if self.Input is not None:
            oldInput = np.copy(self.Input)
        self.Input = inputParameter
        self.forwardProp()
        pred = self.get_predictions()
        self.Input=oldInput
        return pred
    def train(self,batchSize,iterations):
        """
        Forward propogates and back propogates the network to make it do what its supposed to do
        BatchSize - Number of datapoints to train at once
        Iterations - Number of training cycles
        """
        assert batchSize<=self.Labels.size
        TestDataSize = self.Input.shape[1]
        print("Testing Data Size: ",TestDataSize)
        batchesPerEpoch = int(TestDataSize/batchSize)
        print("Batches Per Cycle: ",batchesPerEpoch)
        allTheData = np.copy(self.Input)
        allTheLabels = np.copy(self.Labels)
        allTheAns = np.copy(self.Ans)
        batchData = []
        batchLabels = []
        batchAns = []
        
        scaler= preprocessing.MinMaxScaler()
        for k in range(0,batchesPerEpoch):
            batchData.append(allTheData[:,batchSize*k:min(batchSize*(k+1),TestDataSize)])
            batchLabels.append(allTheLabels[:,batchSize*k:min(batchSize*(k+1),TestDataSize)])
            batchAns.append(allTheAns[:,batchSize*k:min(batchSize*(k+1),TestDataSize)])
        
        
        accuracyData = np.zeros((iterations,3))
        for i in range(iterations):
            
            for j in range(0,len(batchData)):
                self.Input = np.copy(batchData[j])
                self.Input = self.Input+0.125*np.random.rand(self.Input.shape[0],self.Input.shape[1])
                self.Input = scaler.fit_transform(self.Input)
                
                self.Labels = batchLabels[j]
                self.Ans = batchAns[j]
                
                self.forwardProp()
                self.backProp()
            accuracyData[i,0]=i
            accuracyData[i,1]=self.accuracy()
            accuracyData[i,2]=self.test()
            print("Iteration: "+str(i)+" Test Accuracy: "+str(round(accuracyData[i,1],4))+" Validation Acuracy: "+str(round(accuracyData[i,2],4)))
        t=time.gmtime()
        tStr = str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)
        np.savetxt("./output/acuracyLog_"+tStr+".csv",accuracyData,delimiter=",")
        
    
    def test(self):
        """
        Tests the network on a given test set
        """
        #Save to restore after test
        oldInput = np.copy(self.Input)
        oldLabels = np.copy(self.Labels)
        oldAns = np.copy(self.Ans)
        
        self.Input = self.ValidationData
        self.Labels = self.ValidationLabels
        self.Ans = self.ValidationAns
        self.forwardProp()
        #print("Testing Data Acuracy: ",self.accuracy())
        #Restore
        accuracy=self.accuracy()
        self.Input = oldInput
        self.Labels = oldLabels
        self.Ans = oldAns
        return accuracy
