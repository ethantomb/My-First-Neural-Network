
import time
import math
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import NeuralNetwork
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
testAr = np.zeros((2,2))
print(np.where(testAr!=0)[0].size)
#Read in the Train and Test data.        
data = pd.read_csv(r'./input/mnist/mnist_train.csv')
testDatRaw = pd.read_csv(r'./input/mnist/mnist_test.csv')
testDatRaw = np.array(testDatRaw)
nTest,mTest = testDatRaw.shape
#Testing Labels: 1x10000 matrix
TestLabels = testDatRaw.T[0:1]
#Testing Data: 784x10000 matrix
TestData = testDatRaw.T[1:nTest]
#Make all pixels in range 0,1
TestData = TestData/255.
print("Test Data Shape: ",TestData.shape)
print("Test Labels Shape ",TestLabels.shape)

data = np.array(data)
m,n = data.shape

labels = data.T[0:1]
#Training Labels: 1x40000 
labels = labels.T[0:m].T
#Training Pixel Data: 784x40000 normalized between 0,1
Train_Dat = data.T[1:n]
Train_Dat = Train_Dat.T[0:m].T
Train_Dat=Train_Dat/255.
print("Train Labels Shape: ",labels.shape)
print("Train Data Shape: ",Train_Dat.shape)
images = []
#images I made
for im in os.walk('./input/MyTestImages'):
    for i in im[2]:
        
        image = cv2.imread("./input/MyTestImages/"+i,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
        images.append(image/255)

testSet = np.zeros((len(images),784))
for i,image in enumerate(images):
    testI = image.reshape(1,784)
    testSet[i]=testI

myTestSet = testSet.T
#The mnist training data has colored pixels set to 1, white pixels set to 0. 
#For some reason, my data had uncolored pixels set to 1 so I have to invert the image by subtracting 1.
myTestSet = myTestSet-1
myTestSet = np.absolute(myTestSet)
myTestLabels = np.array([0,1,2,4,5,6,7,0,1,2,3,5,7],ndmin=2)
myTestLabels=myTestLabels

#train_where_1 = np.where(Train_Dat>=0.4)
#Train_Dat[train_where_1]=1


time.sleep(2.0)

nNet = NeuralNetwork.NeuralNetwork(Train_Dat,labels,[200,30,10],0.01)
nNet.train(1000,1200)
#nNet.load("./output/Batch500Iter1200")
batchSize=500
iterations = 1200
#nNet.train(batchSize,iterations)
nNet.saveConfiguration("2Hidden1Out")

print("SUCCESS")
print("TrainDat Shape, TestDat Shape: ",Train_Dat.shape,TestData.shape)
print("TrainLabels Shape, TestLabels Shape: ",labels.shape,TestLabels.shape)
predictions = nNet.test(TestData,TestLabels)
#predictions = nNet.test(myTestSet,myTestLabels)

while (1==1):
    i=random.randint(0,len(predictions)-1)
    confidence = np.copy(nNet.HiddenLayers[len(nNet.HiddenLayers)-1].A[:,i])
    confidence=confidence.T
    confidenceSum=0
    k=0
    title= "Correct. " if predictions[i]==nNet.Labels[0,i] else "Incorrect. "
    title+="Number: "+str(nNet.Labels[0,i])+" Guess:"
    while confidenceSum<0.9 and k<5:
        guess=np.argmax(confidence)
        certainty = confidence[guess]
        confidenceSum+=certainty
        title+=str(guess)+": "+str(round(certainty*100,1))+"% "
        confidence[guess]=0
        k+=1
    plt.imshow(nNet.Input.T[i].reshape(28,28))
    plt.title(title)
    plt.show()

