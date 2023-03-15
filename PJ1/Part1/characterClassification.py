import cv2 as cv
import network
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

curPath=os.path.abspath(os.path.dirname(__file__))
trainPath=curPath+"/../train"
def getImg(character:int,number:int)->Tuple[np.ndarray,np.ndarray]:
    imgPath=trainPath+"/"+str(character)+"/"+str(number)+".bmp"
    img=cv.imread(imgPath,0)
    return imgPrepocess(img,character)

def imgPrepocess(img:cv.Mat,character:int)->Tuple[np.ndarray,np.ndarray]:
    ret=np.array(img).flatten()
    out=np.zeros(12)
    out[character-1]=1
    return (ret&1,out)

def showFig(stepArray,lossArray):
    fig = plt.figure()
    plt.plot(stepArray,lossArray)
    ax1 = fig.add_subplot(111)
    l1=ax1.plot(stepArray, lossArray, color='red', label='Loss')
    plt.title('BP Loss')
    plt.legend()
    plt.show()

if __name__=="__main__":
    net=network.network((28**2,100,20,12),0.05,0,True)
    dataSet=[]
    for i in range(1,13):
        for j in range(1,621):
            dataSet.append((i,j))
    random.shuffle(dataSet)
    trainSize=int(620*12*0.9)
    lossArray=[]
    stepArray=[]
    lastLoss=0

    while True:
        b=1
        j=random.randint(1,620)
        input,target=getImg(b,j)
        # j=random.randint(1,trainSize)
        # input,target=getImg(dataSet[j][0],dataSet[j][1])
        input,target=net.prepoccess(input,reshape=True),net.prepoccess(target,reshape=True)
        # output=net.forwardPropagation(input)
        output=net.forwardPropagation(input)
        
        net.backPropagation(target)
        net.step+=1
        
        
        if net.step%1000==0:
            loss=net.crossEntropy(output,target)
            lossArray.append(loss)
            stepArray.append(net.step)
            print("pic={} step={} loss={}".format(j, net.step,loss))
            if abs(lastLoss-loss)<0.000001 and loss < 0.0001 and net.step>10000:
                break
            else:
                lastLoss=loss

    count=0
    for i in range(trainSize+1,12*620):
        input,target=getImg(dataSet[i][0],dataSet[i][1])
        input=net.prepoccess(input,reshape=True)
        output=net.forwardPropagation(input)
        if output.argmax()==dataSet[i][0]-1:
            count+=1
    print("Accuracy:{}".format(count/(12*620-trainSize)))
        
    
    showFig(stepArray,lossArray)