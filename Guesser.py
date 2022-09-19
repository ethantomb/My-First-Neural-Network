#Pygame interface for the network
import sys,pygame
import time
import math
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os
import NeuralNetwork
"""
Loads CSV data from a given path
Expects path to be the location of a .csv file with the top row as labels for the data and the rest as the values. 

"""
def loadData(path):
    raw = np.array(pd.read_csv(path))
    np.random.shuffle(raw)
    m,n = raw.shape
    labels = raw[:,0:1].T
    labels=labels.astype(int)
    data = raw[:,1:n].T
    return data,labels
data = np.array(pd.read_csv("input/doodleData.csv"))
names=None
def initNetwork(doodles):
    if doodles:
        net = NeuralNetwork.NeuralNetwork(np.zeros((784,1)),np.int32(np.zeros((1,1))),[300,80,50,40,11],0.02)
        net.load("output/DoodleParameters/Nov184LayersDoodles",5)      
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
        names = ["airplane","tornado","banana","anvil","door","apple","cat","broom","cactus","couch","fish"]
    else:
        #load numbers data
        net = NeuralNetwork.NeuralNetwork(np.zeros((784,1)),np.int32(np.zeros((1,1))),[300,80,50,40,10],0.02)
        net.load("output/NumberParameters/NumberParameters1_",5)
        names = ["0","1","2","3","4","5","6","7","8","9"]
    return net,names
def getGuess(arr):
    pygame.draw.rect(screen,(128,128,128),[0,640,640,100])
    arr=arr[:,0:640].T
    arr=arr/16777215
    arr=arr-1
    arr=np.absolute(arr)
    YBounds = np.where(arr!=0)[0]
    
    if YBounds.size==0:
        return "-1"
    topY = YBounds[0]
    botY = YBounds[len(YBounds)-1]
    XBounds = np.where(arr!=0)[1]
    
    if XBounds.size==0:
        return "-1"
    
    topX = XBounds[np.argmax(XBounds)]
    botX = XBounds[np.argmin(XBounds)]
    h = botY-topY
    w = botX-topX
    
    arr=arr[max(topY-70,0):min(botY+70,640),max(botX-100,0):min(topX+100,640)]
    
    arr = np.absolute(cv2.resize(arr, dsize=(28,28), interpolation=cv2.INTER_CUBIC))

    
    arr = arr.reshape((784,1))
    arr[np.where(arr>1)]=1
    
    arr = np.absolute(arr)
    predictions = net.classify(arr)

    guess = names[predictions[0]]+": " +str(round(net.HiddenLayers[len(net.HiddenLayers)-1].A[predictions[0]][0]*100,2))+"%"
    
    return guess

net,names = initNetwork(not not not False)

pygame.init()
size = width, height = 640, 740
screen = pygame.display.set_mode(size)
screen.fill((255,255,255))
pygame.draw.rect(screen,(128,128,128),[0,0,320,740])
pygame.draw.rect(screen,(255,128,128),[320,0,320,740])

smallfont = pygame.font.SysFont('Corbel',25)
numersText = smallfont.render('Click Here To Guess Numbers',True,(0,0,0))
doodlesText = smallfont.render('Click Here to Guess Doodles',True,(0,0,0))
text = smallfont.render('L Click: Draw R Click:Erase.' , True , (0,255,255))

smallfont = pygame.font.SysFont('Corbel',35)
prediction=""
ans = smallfont.render(prediction , True , (0,255,255))
networkInitialized = False
draw=True
 
net
names
while True:
    mousePos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    if pygame.mouse.get_pressed()[0]:
        
        if not networkInitialized:
            if(mousePos[0]<320):
                net,names = initNetwork(False)
            else:
                net,names=initNetwork(True)
            networkInitialized=True
            screen.fill((255,255,255))
            pygame.draw.rect(screen,(128,128,128),[0,640,640,100])
            pygame.draw.rect(screen,(255,128,128),[320,640,320,100])
        else:
            if(mousePos[1]<620):
                
                pygame.draw.polygon(screen,(0,0,0),[(mousePos[0],mousePos[1]),(mousePos[0],mousePos[1]+10),(mousePos[0]+4,mousePos[1]),(mousePos[0]+4,mousePos[1]+10)],10)
                
                #mousePos = pygame.mouse.get_pos()
                #event=pygame.event.get()
                
            elif mousePos[1]>=640:    
                if(mousePos[0]<320):
                    screen.fill((255,255,255))
                    pygame.draw.rect(screen,(128,128,128),[0,640,640,100])
                    pygame.draw.rect(screen,(255,128,128),[320,640,320,100])
    elif pygame.mouse.get_pressed()[2] and networkInitialized:
        if(mousePos[1]<620):
            pygame.draw.polygon(screen,(255,255,255),[(mousePos[0],mousePos[1]),(mousePos[0],mousePos[1]+10),(mousePos[0]+4,mousePos[1]),(mousePos[0]+4,mousePos[1]+10)],10)
          
    elif networkInitialized:
        arr = np.array(pygame.surfarray.array2d(screen))
        prediction = getGuess(arr)
    else:
        screen.blit(numersText,(0,320))
        screen.blit(doodlesText,(320,320))
    ans = smallfont.render(prediction , True , (0,255,255))
    
    screen.blit(ans,(0,640)) 
    screen.blit(text,(320,640))
    
    pygame.display.flip()
print("done")

