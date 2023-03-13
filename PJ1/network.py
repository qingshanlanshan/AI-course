from typing import Tuple
import numpy as np

class network:
    def __init__(self, layerNumber:int, nodeNumber:tuple, learningRate:float, decay:float):
        self.learningRate = learningRate
        self.step=0
        self.decay=decay
        # self.target:np.array = np.zeros(outputNodeNumber)
        
        self.layerNumber=layerNumber
        self.layer=[]
        for i in range(layerNumber):
            self.layer.append(np.zeros(nodeNumber[i]))

        self.weight=[]
        self.const=[]
        for i in range(layerNumber-1):
            self.weight.append(np.random.randn(nodeNumber[i+1],nodeNumber[i]))
            self.const.append(np.random.randn(nodeNumber[i+1]))
        
        
        print("init done")
        
    def normalize(self,data:np.array)->np.array:
        maxData = np.max(data)
        minData = np.min(data)
        return (data-minData)/(maxData-minData)    
        
    def denormalize(self, data:np.array,min,max)->np.array:
        minData = np.min(data)
        maxData = np.max(data)
        return data / (maxData - minData) * (max - min) + min
    
    def sigmod(self,x:float)->float:
        return 1/(1+np.exp(-x))
    
    def sigmodDerivative(self,f:float)->float:
        return f*(1-f)
    
    def forwardPropagation(self,input):
        self.layer[0]=input
        for layerNumber in range(self.layerNumber-1):
            self.layer[layerNumber+1]=self.weight[layerNumber].dot(self.layer[layerNumber])+self.const[layerNumber]
            self.layer[layerNumber+1]=self.sigmod(self.layer[layerNumber+1])
        return self.layer[self.layerNumber-1]
    def backPropagation(self,target):
        delta=[None]*(self.layerNumber)
        delta[self.layerNumber-1]=(self.layer[self.layerNumber-1]-target)*self.sigmodDerivative(self.layer[self.layerNumber-1])
        for i in range(self.layerNumber-2,-1,-1):
            delta[i]=np.dot(self.weight[i].T,delta[i+1])
            delta[i]*=self.sigmodDerivative(self.layer[i])
        deltaWeight=[None]*(self.layerNumber-2)
        for i in range(self.layerNumber-1):
            temp=(delta[i+1].reshape(len(delta[i+1]),1)).dot(self.layer[i].reshape(1,len(self.layer[i])))
            self.weight[i]-=self.learningRate*temp
            self.const[i]-=delta[i+1]
            
    def error(self,output,target)->float:
        return np.sum(np.square(target-output))/2
    
    def learningRateDecline(self):
        self.learningRate /= (1+self.decay*self.step)
        

    def train(self,input:np.array, target:np.array)->float:
        input=self.normalize(input)
        target=self.normalize(target)
        self.step+=1
        output=self.forwardPropagation(input)
        self.backPropagation(target)
        self.learningRateDecline()
        return self.error(output,target)
        
    
    
    
        