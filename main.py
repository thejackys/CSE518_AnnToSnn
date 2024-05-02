
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb
from model import ANN, SNN
import numpy as np
train_data = datasets.MNIST(
    root = "data",
    train=True,
    download=True,
    transform=ToTensor(), #Note: it will be scaled to [0,1]
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(X.shape)
    print(y.dtype)
    break

# wandb.init()
device = ("cuda" if torch.cuda.is_available()
          else "cpu")


def train(dataloader, model, loss_func, optimizer, loss=[], ):
    size = len(dataloader.dataset)
    model.train()
    #TODO: Plot the graph
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss:{loss:>3f} [{current:>5d}]/[{size:>5d}]")
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    n_batch = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= n_batch
    correct /= size
    print(f"Test Error:\n Acc: {(100*correct):>0.2f}, Avg loss: {test_loss:>4f}. \n")
    metrics = dict()
    metrics['accuracy'] = correct*100
    metrics['spikes'] = model.get_total_spikes()
    return metrics

def model_run(model='SNN', epochs=5, timepsteps=32, firing_scale = 1.0):
    if model == 'SNN':
        model = SNN(timesteps=timepsteps, firing_scale=firing_scale).to(device)
    else:
        model = ANN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"epoch:{t}\n")
        train(train_dataloader, model, loss_func, optimizer)
        metrics = test(test_dataloader, model, loss_func)
        wandb.log(metrics,step = t)
    return metrics

#reference ANN
# model_run(model='SNN', epochs=20)

#testing T

# T = [1,2,4,8,16,32,64,128,256,512,1024]
# for t in T:
#     metrics = model_run(epochs=5, timepsteps=t)
#     w.log(metrics, step=t)
# print(metrics)

for s in np.linspace(1,0,50,endpoint=False): #scale firing rate from [0.02,...,1]
    run = wandb.init(reinit=True)
    metrics = model_run(epochs=5, timepsteps=32, firing_scale=s)
    run.finish()
    
#testing acc







