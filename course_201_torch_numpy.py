import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print np_data
print torch_data
print tensor2array

data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
print np.abs(data)
print torch.abs(tensor)

print np.sin(data)
print torch.sin(tensor)

print np.mean(data)
print torch.mean(tensor)

data = [[1, 2], [3, 4]]
data = np.array(data)
tensor = torch.FloatTensor(data)

print np.matmul(data, data)
print torch.mm(tensor, tensor)

data = [1, 2]
data = np.array(data)
tensor = torch.FloatTensor(data)

print data.dot(data)
print tensor.dot(tensor)