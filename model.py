
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
import spikingjelly


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.FCNN = nn.Sequential(
            nn.Linear(28*28, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 10, bias=False),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.FCNN(x)
        return logits

class PoissonEncoder(nn.Module):
    def __init__(self, timesteps=100):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, x):
        # Convert input to firing rates
        firing_rates = x #the input is resized to [0,1] already
        batch_size, n_pixels = firing_rates.shape
        T_firing_rates = firing_rates.repeat(1, 1, self.timesteps).reshape(batch_size, self.timesteps, n_pixels)
        # Generate Poisson spikes
        spikes = torch.where(torch.rand_like(T_firing_rates) < T_firing_rates, 1.0, 0.0)
        
        return spikes

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = PoissonEncoder()
        self.layer1 = nn.Linear(28*28, 100, bias=False)
        self.IF = IF_neuron()
        self.layer2 =nn.Linear(100, 10, bias=False)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        
        x = self.layer1(x)
        
        
        logits = self.FCNN(x)
        logits = logits.sum(dim=-1)
        return logits

class IF_neuron(nn.Module):
    '''
    Implementation of the one step Integrate and fire (IF) Neuron. 
    
    :param v_threshold: threshold to fire the neuron
    :type v_threshold: float

    :param v_reset: reset value when the neuron fires
    :type v_threshold: float

    '''
    def __init__(self, v_threshold:float = 1, v_reset:float = 0):
        super().__init__()
        self.v = None
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def fire(self, input):
        """compute each value to see if the spike reset or not"""
        diff = self.v - self.v_threshold 
        return torch.where(diff> 0, 1.0, 0.0)
    def reset(self, spike:torch):
        """
        for the spiked neuron, reset it to v_reset given the 
        """
        self.v = (1.-spike)*self.v
        

    def forward(self, input: Tensor) -> Tensor:
        #init the self.v 
        if self.v is None:
            self.v = torch.full_like(input, self.v_reset)
            print(f'initiating self.v')
        #charge
        self.v = self.v + input
        #spike
        spike = self.fire(input)
        #reset if spike>threshold
        self.reset(spike)
        
        return spike

    def v_float_to_tensor():
        """
        transform the init v if the batch 
        """
    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}'

