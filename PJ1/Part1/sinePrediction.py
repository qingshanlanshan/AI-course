import network
import sampleGeneration as sp
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    net = network.network(nodeNumber=(1, 10, 1), learningRate=0.05)
    curPath=os.path.abspath(os.path.dirname(__file__))
    modulePath=curPath+"/sine.npz"
    net.load(modulePath)
    inputArray=[]
    outputArray=[]
    while True:
        try:
            print("input:")
            x:float=float(input())
            inputArray.append(x)
            y=net.forwardPropagation(net.prepoccess(np.array([x/np.pi]),reshape=True))
            y=y.flatten()*2-1
            print(y)
            outputArray.append(y)
        except KeyboardInterrupt:
            break
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(inputArray, np.sin(np.array(inputArray)), color='red',
                  linestyle='', marker='.', label='DataSet')
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.sin(x)
    l2 = ax1.plot(x, y, color='orange', label='sin(x)')
    l3 = ax1.plot(inputArray, outputArray, color='blue',
                  linestyle='', marker='.', label='Prediction')
    plt.title('BP Prediction')
    plt.legend()
    plt.show()