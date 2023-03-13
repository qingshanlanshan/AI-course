import network
import sample_gen
import numpy as np
import matplotlib.pyplot as plt


def draw_fit_curve(origin_xs, origin_ys, prediction_ys, step_arr, loss_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(origin_xs, origin_ys, color='red', linestyle='', marker='.', label='DataSet')
    x = np.arange(-np.pi,np.pi,0.1)   # start,stop,step
    y = np.sin(x)
    ax1.plot(x,y,color='orange',label='sin(x)')
    ax1.plot(origin_xs, prediction_ys, color='blue', label='Prediction')
    plt.title('BP Prediction')
    ax2 = fig.add_subplot(122)
    ax2.plot(step_arr, loss_arr, color='red', label='Loss')
    plt.title('BP Loss')
    
    

    plt.legend()
    plt.show()
        
        

if __name__=="__main__":
    net = network.network(layerNumber=3,nodeNumber=(100,20,100), learningRate=0.01,decay=0.05)
    x,y=sample_gen.sin_sample_gen_array(100)
    last_error=0
    loss=[]
    step=[]
    while True:
        
        error=net.train(x,y)
        if net.step%1000==0 or net.step==1:
            print("step={} error={}".format(net.step,error))
        loss.append(error)
        step.append(net.step)
        if error-last_error<0.000001 and net.step>10000:
            break
        else:
            last_error=error
        # if KeyboardInterrupt:
        #     break
    output=net.forwardPropagation(net.normalize(x))
    draw_fit_curve(origin_xs=x, origin_ys=y, prediction_ys=net.denormalize(output,-1,1), step_arr=step, loss_arr=loss)

        
    