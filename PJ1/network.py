from typing import Tuple
import numpy as np

class network:
    def __init__(self, inputNodeSize:int, hiddenNodeSize:int, outputNodeSize:int, learningRate:float, decay:float, errorLimit:float):
        self.learningRate = learningRate
        self.step=0
        self.decay=decay
        # self.target:np.array = np.zeros(outputNodeNumber)
        self.errorLimit = errorLimit
        
        self.inputNodeSize = inputNodeSize
        self.hiddenNodeSize = hiddenNodeSize
        self.outputNodeSize = outputNodeSize
        
        self.inputLayer = np.zeros(inputNodeSize)
        self.hiddenLayer = np.zeros(hiddenNodeSize)
        self.outputLayer = np.zeros(outputNodeSize)
        
        self.inputToHiddenWeight = np.random.random((self.inputNodeSize, self.hiddenNodeSize))
        self.hiddenToOutputWeight = np.random.random((self.hiddenNodeSize,self.outputNodeSize))
        
        print("init done")
        
    def normalize(self,data:np.array)->np.array:
        maxData = np.max(data)
        minData = np.min(data)
        return (data-minData)/(maxData-minData)    
        
    def denormalize(self, data:np.array)->np.array:
        minData = np.min(data)
        maxData = np.max(data)
        return data * (maxData - minData) + minData
    
    def sigmod(self,x:float)->float:
        return 1/(1+np.exp(-x))
    
    def sigmodDerivative(self,f:float)->float:
        return f*(1-f)
    
    def inputToHidden(self):
        for i in range(self.inputNodeSize):
            for j in range(self.hiddenNodeSize):
                self.hiddenLayer[j] += self.inputToHiddenWeight[i][j]*self.inputLayer[i]
        for i in range(self.hiddenNodeSize):
            self.hiddenLayer[i] = self.sigmod(self.hiddenLayer[i])
            
    def hiddenToOutput(self):
        for i in range(self.hiddenNodeSize):
            for j in range(self.outputNodeSize):
                self.outputLayer[j] += self.hiddenToOutputWeight[i][j]*self.hiddenLayer[i]
        for i in range(self.outputNodeSize):
            self.outputLayer[i] = self.sigmod(self.outputLayer[i])
            
    def error(self,target)->float:
        return np.sum(np.square(target-self.outputLayer))/2
    
    def learningRateDecline(self):
        self.learningRate /= (1+self.decay*self.step)
        
    def bp(self, target:np.array):
        self.bpInputToHidden(target)
        self.bpHiddenToOutput(target)
    
    def bpHiddenToOutput(self,target):
        for i in range(self.hiddenNodeSize):
            for j in range(self.outputNodeSize):
                self.hiddenToOutputWeight[i][j] += self.learningRate*(target[j]-self.outputLayer[j])*self.sigmodDerivative(self.outputLayer[j])*self.hiddenLayer[i]
    
    def bpInputToHidden(self,target):
        for i in range(self.inputNodeSize):
            for j in range(self.hiddenNodeSize):
                sum=0
                for k in range(self.outputNodeSize):
                    sum+=self.hiddenToOutputWeight[j][k]*(target[k]-self.outputLayer[k])*self.sigmodDerivative(self.outputLayer[k])
                self.inputToHiddenWeight[i][j] += self.learningRate*self.sigmodDerivative(self.hiddenLayer[j])*self.inputLayer[i]*sum
    
    def train(self,input:np.array, target:np.array)->float:
        input=self.normalize(input)
        target=self.normalize(target)
        self.step+=1
        output=self.predict(input)
        self.bp(target)
        self.learningRateDecline()
        return self.error(target)
        
    def predict(self,input:np.array)->np.array:
        self.inputLayer = input
        self.inputToHidden()
        self.hiddenToOutput()
        return self.outputLayer
    
    
    
        