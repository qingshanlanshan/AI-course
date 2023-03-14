import network
import sampleGeneration as sp
import numpy as np
import matplotlib.pyplot as plt


def draw_fit_curve(origin_xs, origin_ys, prediction_ys, step_arr, loss_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    l1 = ax1.plot(origin_xs, origin_ys, color='red',
                  linestyle='', marker='.', label='DataSet')
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.sin(x)
    l2 = ax1.plot(x, y, color='orange', label='sin(x)')
    l3 = ax1.plot(origin_xs, prediction_ys, color='blue',
                  linestyle='', marker='.', label='Prediction')
    plt.title('BP Prediction')
    plt.legend()

    ax2 = fig.add_subplot(122)
    l4 = ax2.plot(step_arr, loss_arr, color='red', label='Loss')
    plt.title('BP Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    net = network.network(nodeNumber=(1, 20, 1), learningRate=0.05, decay=0)
    x, y = sp.sin_sample_gen_array(1000000,sort=False)
    yBar=(y+1)/2
    last_error = 0
    loss = []
    step = []
    for i in range(len(x)):
        a = x[i]
        b = yBar[i]
        error = net.train(np.array([a]), np.array([b]), False)
        if net.step % 1000 == 0 or net.step == 1:
            print("step={} error={}".format(net.step, error))
            loss.append(error)
            step.append(net.step)
        # if error-last_error<0.0000001 and net.step>10000 and error<0.001:
        #     break
        # else:
        #     last_error=error
        # if KeyboardInterrupt:
        #     break
    output = []
    for i in x:
        output.append(net.predict(np.array([i]), False)[0])
    draw_fit_curve(origin_xs=x, origin_ys=y,
                   prediction_ys=np.array(output)*2-1, step_arr=step, loss_arr=loss)
