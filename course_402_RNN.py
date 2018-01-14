import torch
import torch.nn as NN
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

# print train_data.train_data.size()
# print train_data.train_labels.size()
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor) / 255.0
test_y = test_data.test_labels

class RNN(NN.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = NN.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True, # [batch_size, time_step, input_size]
        )
        self.out = NN.Linear(in_features=64, out_features=10)
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        # out = torch.mean(self.out(r_out), dim=1)
        return out

rnn = RNN()
# print rnn

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = NN.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.view(-1, 28, 28))
        batch_y = Variable(batch_y)

        output = rnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print 'Epoch={}, step={}, train loss={}, test accuracy={}'.format(epoch, step, loss.data[0], accuracy)

test_output = rnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print 'prediction number={}'.format(pred_y)
print 'real number={}'.format(test_y[:10].numpy())