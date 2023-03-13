import network
import sample_gen
import numpy as np

if __name__=="__main__":
    net = network.network(inputNodeSize=100, hiddenNodeSize=10, outputNodeSize=100, learningRate=0.01,decay=0.5, errorLimit=0.000001)
    
    while True:
        x,y=sample_gen.sin_sample_gen_array(100)
        error=net.train(x[:, np.newaxis],y[:, np.newaxis])
        print("step={} error={}".format(net.step,error))
        if error<0.000000001 and net.step>100000:
            break
        # if KeyboardInterrupt:
        #     break
        
    # while True:
    #     print("input: ")
    #     x=float(input())
    #     net.predict(np.array([x/np.pi]))
    #     target=np.sin(x)
    #     print("target={} output={} error={}%".format(target,net.outputLayer[0],(net.outputLayer[0]-target)/target*100))
        
        