import torch
import torch.nn as NN
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

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
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor) / 255.0
test_y = test_data.test_labels

class CNN(NN.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = NN.Sequential(
            NN.Conv2d( # (1, 28, 28)
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2 # if stride=1, padding=(kernel_size-1)/2
            ), # -> (32, 28, 28)
            NN.ReLU(), # -> (32, 28, 28)
            NN.MaxPool2d(kernel_size=2) # -> (32, 14, 14)
        )
        self.conv2 = NN.Sequential(
            NN.Conv2d( # (32, 14, 14)
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2  # if stride=1, padding=(kernel_size-1)/2
            ), # -> (64, 14, 14)
            NN.ReLU(), # -> (64, 14, 14)
            NN.MaxPool2d(kernel_size=2) # -> (64, 7, 7)
        )
        self.out = NN.Linear(in_features=64*7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) # (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1) # -> (batch_size, 64*7*7)
        output = self.out(x)
        return output

cnn = CNN()
# print cnn

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = NN.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)

        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print 'Epoch={}, step={}, train loss={}, test accuracy={}'.format(epoch, step, loss.data[0], accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print 'prediction number={}'.format(pred_y)
print 'real number={}'.format(test_y[:10].numpy())