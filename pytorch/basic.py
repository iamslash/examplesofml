# regression ANN
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4])
y = 2 * x + 1
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

class ANN_regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN_regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x)            :
        return self.linear(x)

def main(epochs=2000):
    # Create instance of model
    model = ANN_regression(1, 1)
    criterion = nn.MSELoss()
    learning_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        epoch += 1
        inputs = Variable(torch.from_numpy(x[:3]))
        labels = Variable(torch.from_numpy(y[:3]))

        optimiser.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        
        if epoch % 100 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.data[0]))

    # Print Predictions
    predicted = model.forward(Variable(torch.from_numpy(x[3:])))
    plt.plot(x, y, 'go', label = 'targets', alpha = 0.5)
    plt.plot(x, predicted, label = 'predictions', alpha = 0.5)
    plt.show()
    print(model.state_dict())

if __name__ == '__main__':
    main()
