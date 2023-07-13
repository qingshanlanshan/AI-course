import characterClassifierTrain as cc
import network
import os
from typing import Tuple
import numpy as np
import cv2 as cv

curPath=os.path.abspath(os.path.dirname(__file__))
trainPath="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/test_data"

def imgPrepocess(img:cv.Mat,character:int)->Tuple[np.ndarray,np.ndarray]:
    ret=np.array(img).flatten()
    out=np.zeros(12)
    out[character-1]=1
    return (ret&1,out)

def test(net:network.network,testSet:list):
    count=0
    for i in testSet:
        input,target=getImg(i[0],i[1])
        input=net.prepoccess(input,reshape=True)
        output=net.forwardPropagation(input)
        if output.argmax()==i[0]-1:
            count+=1
    return count/len(testSet)

def getImg(character:int,number:int)->Tuple[np.ndarray,np.ndarray]:
    imgPath=trainPath+"/"+str(character)+"/"+str(number)+".bmp"
    img=cv.imread(imgPath,0)
    return imgPrepocess(img,character)

if __name__=="__main__":

    net=network.network((28**2,100,20,12),0.1,softmax=True)
    DataSet=[]
    for i in range(1,13):
        for j in range(1,241):
            DataSet.append((i,j))
    net.load(curPath+"/characterClassifier-80.npz")
    accuracy=test(net,DataSet)
    print("Accuracy={}%".format(100.*accuracy))
        
        


