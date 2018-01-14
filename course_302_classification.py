import torch
from torch.autograd import Variable
import torch.nn.functional as Function
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(means=2*n_data, std=1)
y0 = torch.zeros(100)
x1 = torch.normal(means=-2*n_data, std=1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)
x, y = Variable(x), Variable(y)

# plt.scatter(x=x.data.numpy()[:, 0], y=x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_features=n_feature, out_features=n_hidden)
        self.predict = torch.nn.Linear(in_features=n_hidden, out_features=n_output)
    def forward(self, x):
        x = Function.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)
print net

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 10 == 0 or t in [3, 6]:
        plt.cla()
        prediction = torch.max(Function.softmax(out), 1)[1]
        # pred_y = prediction.data.numpy().squeeze()
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x=x.data.numpy()[:, 0], y=x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.0
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.show()
        plt.pause(0.2)

plt.ioff()
