
import torch as t
from matplotlib import pyplot as plt

t.manual_seed(1)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size,1)*20
    y = x*2+(1+t.randn(batch_size,1))*3
    return x,y

x,y = get_fake_data(20)
a = plt.scatter(x.numpy(),y.numpy())
plt.show(a)

w = t.rand(1,1)
b = t.zeros(1,1)
lr = 0.001
