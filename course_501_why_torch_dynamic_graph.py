import torch
import torch.nn as NN
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target(cos)')
# plt.plot(steps, x_np, 'b-', label='input(sin)')
# plt.legend(loc='best')
# plt.show()

class RNN(NN.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = NN.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = NN.Linear(in_features=32, out_features=1)
    def forward(self, x, h_state):
        '''
            x -> [batch_size, time_step, input_size]
            r_out -> [batch_size, time_step, hidden_size]
            h_state -> [num_layers, batch, hidden_size]
        '''
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
# print rnn

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = NN.MSELoss()

h_state = None
plt.figure(1, figsize=(12, 5))
plt.ion()
for step in range(100):

    dynamic_steps = np.random.randint(1, 4)

    start, end = step * np.pi, (step+dynamic_steps) * np.pi
    step += dynamic_steps
    steps = np.linspace(start, end, 10*dynamic_steps, dtype=np.float32)

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data) # !!!

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()