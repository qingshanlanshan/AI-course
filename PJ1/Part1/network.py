from typing import Tuple
import numpy as np

class network:
    def __init__(self, nodeNumber:tuple, learningRate:float,softmax=False,dropout=[]):
        self.learningRate = learningRate
        self.step=0
        # self.target:np.array = np.zeros(outputNodeNumber)
        self.enableSoftmax=softmax
        self.dropout={}
        for i in dropout:
            self.dropout[i[0]]=i[1]
        self.mask={}
        self.nodeNumber=nodeNumber
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
    
    def sigmod(self,x:np.ndarray)->np.ndarray:
        return 1/(1+np.exp(-x))
    
    def sigmodDerivative(self,f:np.ndarray)->np.ndarray:
        return f*(1-f)
    
    def forwardPropagation(self,input:np.ndarray)->np.ndarray:
        self.mask={}
        self.layer[0]=input
        for layerNumber in range(self.layerNumber-2):
            self.layer[layerNumber+1]=self.weight[layerNumber].dot(self.layer[layerNumber])+self.const[layerNumber]
            self.layer[layerNumber+1]=self.sigmod(self.layer[layerNumber+1])
            if layerNumber+1 in self.dropout:
                mask=np.random.binomial(1/(1-self.dropout[layerNumber+1]),1-self.dropout[layerNumber+1],size=self.layer[layerNumber+1].shape)
                self.mask[layerNumber+1]=mask
                self.layer[layerNumber+1]*=mask
        self.layer[-1]=self.weight[-1].dot(self.layer[-2])+self.const[-1]
        if self.enableSoftmax:
            self.layer[-1]=self.softmax(self.layer[-1])
        else:
            self.layer[-1]=self.sigmod(self.layer[-1])
        return self.layer[-1]

    def backPropagation(self,target:np.ndarray):
        delta=[None]*(self.layerNumber)
        if self.enableSoftmax:
            delta[self.layerNumber-1]=(self.layer[-1]-target)/len(target)
        else:
            delta[self.layerNumber-1]=(self.layer[self.layerNumber-1]-target)*self.sigmodDerivative(self.layer[self.layerNumber-1])
        for i in range(self.layerNumber-2,0,-1):
            delta[i]=np.dot(self.weight[i].T,delta[i+1])*self.sigmodDerivative(self.layer[i])
            # if i in self.mask:
            #     delta[i]*=self.mask[i]
        for i in range(self.layerNumber-1):
            if i+1 in self.mask:
                self.weight[i]-=self.learningRate*(delta[i+1]*self.mask[i+1]).dot(self.layer[i].T)
            else:
                self.weight[i]-=self.learningRate*delta[i+1].dot(self.layer[i].T)
            self.const[i]-=self.learningRate*delta[i+1]
        
            
    def error(self,output,target)->float:
        return np.sum(np.square(target-output))/2
    
    def crossEntropy(self,output:np.ndarray,target:np.ndarray)->float:
        return -np.average(target*np.log(output))
    
    def learningRateDecline(self):
        self.learningRate /= (1+self.decay)
        

    def train(self,input:np.ndarray, target:np.ndarray)->float:
        self.step+=1
        output=self.forwardPropagation(input)
        self.backPropagation(target)
        self.learningRateDecline()
        return self.error(output,target)
        
    def prepoccess(self,input:np.ndarray,normalize:bool=False,reshape=False)->np.ndarray:
        if normalize:
            input=self.normalize(input)
        if reshape:
            input=input.reshape((len(input),1))
        return input
    
    def dump(self,filename:str):
        np.savez(filename, x=np.array(self.weight, dtype=object),y=np.array(self.const, dtype=object), allow_pickle=True)
        
    def load(self,filename:str):
        file=np.load(filename, allow_pickle=True)
        self.weight=file["x"]
        self.const=file["y"]
        
    def softmax(self,x:np.ndarray)->np.ndarray:
        return np.exp(x)/np.sum(np.exp(x))