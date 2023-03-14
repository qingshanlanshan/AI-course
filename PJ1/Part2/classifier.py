import torch
import torch.nn.functional as function
import
   
# 方法1，通过定义一个Net类来建立神经网络
class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(n_feature, n_hidden)
    self.predict = torch.nn.Linear(n_hidden, n_output)
   
  def forward(self, x):
    x = function.sigmoid(self.hidden(x))
    x = self.predict(x)
    return x
   
net1 = Net(2, 12, 1)

if __name__=="__init__":
    inputData=