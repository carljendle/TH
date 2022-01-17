from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(10)
N = 30

x = torch.randn(N,1)
print(x)
#Ger oss ett linjärt samband mellan x och y, dock med lite brus.
y = x + torch.randn(N,1)/2

model = nn.Sequential(nn.Linear(1,1), nn.ReLU(), nn.Linear(1,1))

print(model)


learning_rate = 0.05

#MSE - vi vill predicta ett kontinuerligt värde i och med regressionsmodell
loss_criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_epochs = 100

#Vi kan lägga loss i lista eller tensor - vilken iterable som helst fungerar.
losses = torch.zeros(n_epochs)
list_losses = []

for epoch in range(n_epochs):

    predictions = model(x)

    loss = loss_criterion(predictions, y)

    losses[epoch] = loss
    list_losses.append(loss.item())
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



preds = model(x)

#Detachsteget gör att vi kopplar loss från gradientberäkningarna. Vill vi använda 
print(preds.detach())

testloss = ((preds - y)**2).mean()

plt.plot(losses.detach(), 'o')
plt.plot(n_epochs,testloss.detach(), 'ro')
plt.show()
