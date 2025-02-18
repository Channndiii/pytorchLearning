import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(12)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x), Variable(y)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=10),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=10, out_features=1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    torch.save(net1, 'net.pkl')
    torch.save(net1.state_dict(), 'net_params.pkl')

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=10),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=10, out_features=1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))

    prediction = net3(x)

    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

save()
restore_net()
restore_params()