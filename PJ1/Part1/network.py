from typing import Tuple
import numpy as np

class network:
    def __init__(self, nodeNumber:tuple, learningRate:float, decay:float):
        self.learningRate = learningRate
        self.step=0
        self.decay=decay
        # self.target:np.array = np.zeros(outputNodeNumber)
        
        self.layerNumber=len(nodeNumber)
        self.layer=[]
        for i in range(self.layerNumber):
            self.layer.append(np.zeros((nodeNumber[i],1)))

        self.weight=[]
        self.const=[]
        for i in range(self.layerNumber-1):
            self.weight.append(np.random.uniform(-1, 1, (nodeNumber[i+1],nodeNumber[i]))/np.sqrt(nodeNumber[i]))
            self.const.append(-np.random.uniform(-1,0,size=(nodeNumber[i+1],1)))
        self.const[-1]=np.random.uniform(-0.2,0.2,(nodeNumber[-1],1))
        
        
        print("init done")
        
    def normalize(self,data:np.ndarray)->np.ndarray:
        maxData = np.max(data)
        minData = np.min(data)
        return (data-minData)/(maxData-minData)    
        
    def denormalize(self, data:np.ndarray,min,max)->np.ndarray:
        minData = np.min(data)
        maxData = np.max(data)
        return data / (maxData - minData) * (max - min) + min
    
    def sigmod(self,x:float)->float:
        return 1/(1+np.exp(-x))
    
    def sigmodDerivative(self,f:float)->float:
        return f*(1-f)
    
    def forwardPropagation(self,input:np.ndarray):
        self.layer[0]=input
        for layerNumber in range(self.layerNumber-1):
            self.layer[layerNumber+1]=self.weight[layerNumber].dot(self.layer[layerNumber])+self.const[layerNumber]
            self.layer[layerNumber+1]=self.sigmod(self.layer[layerNumber+1])
        return self.layer[self.layerNumber-1]
    
    def backPropagation(self,target:np.ndarray):
        delta=[None]*(self.layerNumber)
        delta[self.layerNumber-1]=(self.layer[self.layerNumber-1]-target)*self.sigmodDerivative(self.layer[self.layerNumber-1])
        for i in range(self.layerNumber-2,0,-1):
            delta[i]=np.dot(self.weight[i].T,delta[i+1])*self.sigmodDerivative(self.layer[i])
        
        for i in range(self.layerNumber-1):
            self.weight[i]-=self.learningRate*delta[i+1].dot(self.layer[i].T)
            self.const[i]-=self.learningRate*delta[i+1]
            
    def error(self,output,target)->float:
        return np.sum(np.square(target-output))/2
    
    def learningRateDecline(self):
        self.learningRate /= (1+self.decay)
        

    def train(self,input:np.ndarray, target:np.ndarray,Normalize:bool=True)->float:
        if Normalize:
            input=self.normalize(input)
            target=self.normalize(target)
        input=input.reshape((len(input),1))
        target=target.reshape((len(target),1))
        self.step+=1
        output=self.forwardPropagation(input)
        self.backPropagation(target)
        self.learningRateDecline()
        return self.error(output,target)
        
    
    def predict(self,input:np.ndarray,Normalize:bool=True)->np.ndarray:
        if Normalize:
            input=self.normalize(input)
        input=input.reshape((len(input),1))
        output=self.forwardPropagation(input)
        output=output.reshape(len(output))
        if Normalize:
            output=self.denormalize(output,-1,1)
        return output
    
        