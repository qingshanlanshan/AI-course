import characterClassifierTrain as cc
import network
import os

if __name__=="__main__":
    curPath=os.path.abspath(os.path.dirname(__file__))
    trainPath=curPath+"/../train"
    net=network.network((28**2,100,20,12),0.1,softmax=True)
    DataSet=[]
    for i in range(1,13):
        for j in range(1,int(620*0.9+1)):
            DataSet.append((i,j))
    net.load(curPath+"/characterClassifier.npz")
    accuracy=cc.test(net,DataSet)
    print("Accuracy={}%".format(100.*accuracy))
        
        


