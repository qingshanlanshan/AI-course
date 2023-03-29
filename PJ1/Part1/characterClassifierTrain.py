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

def showFig(stepArray,lossArray,accuracyArray):
    fig = plt.figure()
    # plt.plot(stepArray,lossArray)
    ax1 = fig.add_subplot(121)
    l1=ax1.plot(stepArray, lossArray, color='red', label='Loss')
    plt.title('BP Loss')
    plt.legend()
    
    ax2=fig.add_subplot(122)
    l2=ax2.plot(stepArray,accuracyArray,color='blue',label='Accuracy')
    plt.title('BP Accuracy')
    plt.legend()
    plt.show()
    
def test(net:network.network,testSet:list):
    count=0
    for i in testSet:
        input,target=getImg(i[0],i[1])
        input=net.prepoccess(input,reshape=True)
        output=net.forwardPropagation(input)
        if output.argmax()==i[0]-1:
            count+=1
    return count/len(testSet)

if __name__=="__main__":
    net=network.network((28**2,100,20,12),0.1,softmax=True)
    trainSet=[]
    for i in range(1,13):
        for j in range(1,int(620*0.9+1)):
            trainSet.append((i,j))
    testSet=[]
    for i in range(1,13):
        for j in range(int(620*0.9+1),621):
            trainSet.append((i,j))
    trainSize=int(620*12*0.9)
    lossArray=[]
    stepArray=[]
    accuracyArray=[]
    lastLoss=0
    epoch=0
    while True:
        try:
            random.shuffle(trainSet)
            for item in trainSet:
                input,target=getImg(item[0],item[1])
                # j=random.randint(1,trainSize)
                # input,target=getImg(dataSet[j][0],dataSet[j][1])
                input,target=net.prepoccess(input,reshape=True),net.prepoccess(target,reshape=True)
                # output=net.forwardPropagation(input)
                output=net.forwardPropagation(input)
                net.backPropagation(target)
                # net.step+=1
                
                
            epoch+=1
            # net.learningRate/=(1+0.001)
            loss=net.crossEntropy(output,target)
            accuracy=test(net,trainSet)
            lossArray.append(loss)
            # stepArray.append(net.step)
            stepArray.append(epoch)
            accuracyArray.append(accuracy)
            # print("step={} loss={} accuracy={}".format(net.step,loss,accuracy))
            print("epoch={} loss={} accuracy={}".format(epoch,loss,accuracy))
            if abs(lastLoss-loss)<0.000001 and loss < 0.00001 and accuracy>0.9999:
                break
            else:
                lastLoss=loss
        except KeyboardInterrupt:
            break
    # net.dump(curPath+"/characterClassifier.npz")
    showFig(stepArray,lossArray,accuracyArray)
                

    