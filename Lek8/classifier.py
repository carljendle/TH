import torch
import torch.nn as nn
import numpy as np



#Olika centroider för datan
A = [1, 1]
B = [5, 1]

n_per_clust = 100

#Gör exempel med lite random noise
a = [A[0] + np.random.randn(n_per_clust), A[1] + np.random.randn(n_per_clust)]
b = [B[0] + np.random.randn(n_per_clust), B[1] + np.random.randn(n_per_clust)]

labels = np.vstack((np.zeros((n_per_clust,1)), np.ones((n_per_clust,1))))
print(labels.shape)

data = np.hstack((a,b)).T

print(data.shape)


data = torch.tensor(data).float()
labels = torch.tensor(labels).float()

model = nn.Sequential(
    nn.Linear(2,1),
    nn.ReLU(),
    nn.Linear(1,1),
    nn.Sigmoid()
)
#Överblick av modellen
print(model)

learning_rate = 0.01

loss_criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


num_epochs = 1000

losses = torch.zeros(num_epochs)


for epoch in range(num_epochs):

    #Forward pass
    preds = model(data)

    #Beräkna och logga loss
    loss = loss_criterion(preds, labels)
    losses[epoch] = loss

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


preds = model(data).detach()
#print(preds.detach())
for pred in preds:
    print(pred)
