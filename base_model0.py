import torch
import numpy as np
import matplotlib.pyplot as plt


ENCODER_DIMENSIONALITY = 25
#def encoder(x):
#    return torch.tensor([x**i / 10**max(0, i-3) for i in range(ENCODER_DIMENSIONALITY)]).cpu().T


class LinearFeatureModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearFeatureModel, self).__init__()
        self.input_dim = input_dim
        self.linearTransform = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linearTransform(x)

class BetterModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(BetterModel, self).__init__()
        self.first = torch.nn.Linear(1, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.last = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.first(x)
        x = self.relu(x)
        x = self.last(x)
        return x

x = np.concatenate((np.linspace(0, 6.28, 100), np.linspace(6.28, 12.56, 10)), axis=0)
y = np.sin(x)

x_test = np.linspace(0, 12.56, 100)
x_test = torch.tensor(x_test).unsqueeze(dim=-1)
y_test = np.sin(x_test)
#feature_test = encoder(x_test)
y_test = torch.tensor(y_test)

#plt.plot(x, y)
#plt.show()

#feature_vector = encoder(x)
#print(feature_vector.shape)
x = torch.tensor(x).unsqueeze(dim=-1)
print(x.shape)
y = torch.tensor(y)
print(y.shape)
model = BetterModel(ENCODER_DIMENSIONALITY)
#preds = model(feature_vector.float())

#torch.nn.functional.mse_loss(preds.squeeze(dim=-1), y)

training_iters = 30000
optimizer = torch.optim.Adam(model.parameters())
model.train()

for i in range(training_iters):
    #full batch training
    optimizer.zero_grad()
    preds = model(x.float())
    loss = torch.nn.functional.mse_loss(preds.squeeze(dim=-1), y.float())
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print(loss.item())

with torch.no_grad():
    model.eval()
    output_y = model(x_test.float()).numpy()
    plt.plot(x_test, output_y, "r")
    plt.plot(x_test, y_test.numpy(), "g")
    plt.show()
