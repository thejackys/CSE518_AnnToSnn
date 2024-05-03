
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
    if isinstance(model, SNN):
        metrics['spikes'] = model.get_total_spikes()
    return metrics
def model_run(model='SNN', epochs=5, timepsteps=32, firing_scale = 1.0,
            add_constraint=False,
            do_spike_norm=False,
            do_surrogate=False):
    if model == 'SNN':
        model = SNN(timesteps=timepsteps, 
                    firing_scale=firing_scale, 
                    do_spike_norm=do_spike_norm,
                    do_surrogate=do_surrogate).to(device)
    else:
        model = ANN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"epoch:{t}\n")
        train(train_dataloader, model, loss_func, optimizer)
        if add_constraint:
            model.apply_weight_constraints()
        metrics = test(test_dataloader, model, loss_func)
        wandb.log(metrics,step = t)
    return metrics

def experiment(do_spike_norm, do_surrogate):
    
    # find best T for firing_rate=1

    T = [1,2,4,8,16,32,64,128,256,512]
    best_acc = 0
    best_T = 0
    for t in T:  
        run = wandb.init(reinit=True)
        metrics = model_run(epochs=5, timepsteps=t,
                            add_constraint=False,
                            do_spike_norm=do_spike_norm,
                            do_surrogate=do_surrogate)
        if metrics['accuracy'] > best_acc:
            best_T = t
            best_acc = metrics['accuracy']

        run.log({"timesteps":t})
        run.finish()
    print(f"best metrics found in timesteps = {best_T}, with acc = {best_acc}.")

    #for best T, find best firing_rate s

    for s in np.linspace(1,0,50,endpoint=False): #scale firing rate from [0.02,...,1]
        run = wandb.init(reinit=True)
        metrics = model_run(epochs=5, timepsteps=best_T, firing_scale=s,
                            add_constraint=False,
                            do_spike_norm=do_spike_norm,
                            do_surrogate=do_surrogate)
        run.log({"firing_rate":s})
        run.finish()


#reference ANN
run = wandb.init(reinit=True)
model_run(model='ANN', epochs=5)
run.finish()

#################
########no spiking norm
###################

experiment(do_spike_norm=False, do_surrogate=False)

#################
########spiking norm
###################
experiment(do_spike_norm=True, do_surrogate=False)


#################
########no spiking norm w/surrogate
###################
experiment(do_spike_norm=False, do_surrogate=True)

#################
########spiking norm w/surrogate
###################
experiment(do_spike_norm=True, do_surrogate=True)



# metrics = model_run(epochs=16, timepsteps=best_T, firing_scale=?, add_constraint=True, do_spike_norm=True)    
# metrics = model_run(epochs=16, timepsteps=best_T, firing_scale=?, add_constraint=True, do_spike_norm=True)






