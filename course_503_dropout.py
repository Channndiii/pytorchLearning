import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

N_SAMPLES = 20
N_HIDDEN = 300

x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
x, y = Variable(x), Variable(y)

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
test_x, test_y = Variable(test_x, volatile=True), Variable(test_y, volatile=True)

# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=N_HIDDEN, out_features=N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=N_HIDDEN, out_features=1)
)

net_dropout = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=N_HIDDEN, out_features=N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=N_HIDDEN, out_features=1)
)

# print net_overfitting
# print net_dropout

optimizer_oft = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_dpt = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()
for i in range(500):
    pred_oft = net_overfitting(x)
    pred_dpt = net_dropout(x)
    loss_oft = loss_func(pred_oft, y)
    loss_dpt = loss_func(pred_dpt, y)

    optimizer_oft.zero_grad()
    optimizer_dpt.zero_grad()
    loss_oft.backward()
    loss_dpt.backward()
    optimizer_oft.step()
    optimizer_dpt.step()

    if i % 10 == 0:
        net_overfitting.eval()
        net_dropout.eval() # !!!

        plt.cla()

        test_pred_oft = net_overfitting(test_x)
        test_pred_dpt = net_dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_oft.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_dpt.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_oft, test_y).data[0], fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_dpt, test_y).data[0], fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        net_overfitting.train() # !!!
        net_dropout.train() # !!!

plt.ioff()
