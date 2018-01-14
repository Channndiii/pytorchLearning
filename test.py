import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

x = np.reshape(np.asarray([i for i in range(24)], dtype=np.float32), [3, 2, 4])
x = Variable(torch.from_numpy(x))
print x
x = x.transpose(1, 2)
print x

net = nn.Sequential(
    nn.BatchNorm1d(num_features=4, affine=False)
)

result = net(x)
result = result.transpose(1, 2)
print result.data.numpy()