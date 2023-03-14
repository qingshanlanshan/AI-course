import cv2 as cv
import network
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

curPath=os.path.abspath(os.path.dirname(__file__))
trainPath=curPath+"/../train"
def getImg(character:int,number:int):
    img=trainPath+"/"+str(character)+"/"+str(number)+".bmp"
    return cv.imread(img,0)

def prepocessing(img,i):
    ret=np.array(img).flatten()
    out=np.zeros(12)
    out[i-1]=1
    return ret&1,out

def showFig(stepArray,lossArray):
    fig = plt.figure()
    plt.plot(stepArray,lossArray)
    ax1 = fig.add_subplot(111)
    l1=ax1.plot(stepArray, lossArray, color='red', label='Loss')
    plt.title('BP Loss')
    plt.legend()
    plt.show()

if __name__=="__main__":
    net=network.network(3,(28**2,50,12),0.1,0.05)
    dataSet=[]
    for i in range(1,13):
        for j in range(1,621):
            dataSet.append((i,j))
    random.shuffle(dataSet)
    trainSize=int(620*12*0.9)
    lossArray=[]
    stepArray=[]
    lastLoss=0
    for i in range(trainSize):
        img=getImg(dataSet[i][0],dataSet[i][1])
        input,output=prepocessing(img,dataSet[i][0])
        loss=net.train(input,output,True)
        if net.step%1000==0:
            lossArray.append(loss)
            stepArray.append(net.step)
            print("i={} step={} loss={}".format(i, net.step,loss))
        if abs(lastLoss-loss)<0.000001 and loss < 0.00001:
            break
        else:
            lastLoss=loss

    count=0
    for i in range(trainSize+1,12*620):
        img=getImg(dataSet[i][0],dataSet[i][1])
        input,output=prepocessing(img,dataSet[i][0])
        output=net.forwardPropagation(input)
        if output.argmax()==dataSet[i][0]-1:
            count+=1
    print("Accuracy:{}".format(count/(12*620-trainSize)))
        
    
    showFig(stepArray,lossArray)