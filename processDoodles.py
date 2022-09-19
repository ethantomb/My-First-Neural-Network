#Processes the npy files of doodles and saves them as a csv file
import ndjson
import numpy as np
import sys
import os
import json
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
"""
0-airplane
1-tornado
2-banana
3-anvil
4-door
5-apple
6-cat
7-broom
8-cactus
9-couch
10-fish
"""

airplane = np.load("npy/full_numpy_bitmap_airplane.npy")
tornado = np.load("npy/full_numpy_bitmap_tornado.npy")
banana = np.load("npy/full_numpy_bitmap_banana.npy")
anvil = np.load("npy/full_numpy_bitmap_anvil.npy")
door = np.load("npy/full_numpy_bitmap_door.npy")
apple = np.load("npy/full_numpy_bitmap_apple.npy")
cat = np.load("npy/full_numpy_bitmap_cat.npy")
broom = np.load("npy/full_numpy_bitmap_broom.npy")
cactus = np.load("npy/full_numpy_bitmap_cactus.npy")
couch = np.load("npy/full_numpy_bitmap_couch.npy")
fish = np.load("npy/full_numpy_bitmap_fish.npy")

nTest = 5000
nTrain = 500

def saveData():
    data = np.concatenate((airplane[0:nTest],tornado[0:nTest],banana[0:nTest],anvil[0:nTest],door[0:nTest],apple[0:nTest],cat[0:nTest],broom[0:nTest],cactus[0:nTest],couch[0:nTest],fish[0:nTest]),axis=0)
    labels = np.concatenate((np.zeros((nTest,1)),np.ones((nTest,1)),np.full((nTest,1),2),np.full((nTest,1),3),np.full((nTest,1),4),np.full((nTest,1),5),np.full((nTest,1),6),np.full((nTest,1),7),np.full((nTest,1),8),np.full((nTest,1),9),np.full((nTest,1),10)),axis=0)
    print(data.shape)
    data = np.concatenate((labels,data),axis=1)
    
    np.savetxt("input/doodleData.csv",data,delimiter=",")
    print("Data saved to doodleData.csv")
#take the next 2000 samples from each with their labels 0-10 as the first row. Save as a csv file for testing
def saveTestData():
    data = np.concatenate((airplane[nTest:nTest+nTrain],tornado[nTest:nTest+nTrain],banana[nTest:nTest+nTrain],anvil[nTest:nTest+nTrain],door[nTest:nTest+nTrain],apple[nTest:nTest+nTrain],cat[nTest:nTest+nTrain],broom[nTest:nTest+nTrain],cactus[nTest:nTest+nTrain],couch[nTest:nTest+nTrain],fish[nTest:nTest+nTrain]),axis=0)
    labels = np.concatenate((np.zeros((nTrain,1)),np.ones((nTrain,1)),np.full((nTrain,1),2),np.full((nTrain,1),3),np.full((nTrain,1),4),np.full((nTrain,1),5),np.full((nTrain,1),6),np.full((nTrain,1),7),np.full((nTrain,1),8),np.full((nTrain,1),9),np.full((nTrain,1),10)),axis=0)
    data = np.concatenate((labels,data),axis=1)
    np.savetxt("input/doodleTestData.csv",data,delimiter=",")
    print("Data saved to doodleTestData.csv")
saveData()
bigAssVariable = np.array(pd.read_csv("input/doodleData.csv",header=None))
print(bigAssVariable.shape)

plt.imshow(bigAssVariable[0,1:].reshape(28,28))
plt.show()

    

