
import spikingjelly.activation_based
import spikingjelly.activation_based.neuron
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
import spikingjelly
def net_reinitialize(net:nn.Module):
    for m in net.modules():
        if hasattr(m, 'reinitialize'):
            #reinitialize the memristor value
            m.reinitialize()

def quantize_weights(weights:torch.tensor, num_bits:int =4):
    '''
        Implementation of the weight clustering. It will cluster the weights into 32 bins: 16 bins for each positive and negative values. The weights is not necessarly an integer. 
        
        :param weights: weights to be shared 
        :type timesteps: torch.tensor

        :param num_bits: decide the numbers of shared weights. 2 ^ num_bits
        :type num_bits: int

        :return weights: clusterd weight values
        :type return: torch.tensor
        
    '''
    num_levels = 2 ** num_bits
    
    # Quantize positive weights
    pos_weights = weights[weights > 0]
    if len(pos_weights) > 0:
        max_val = pos_weights.max()
        min_val = pos_weights.min()
        step = (max_val - min_val) / (num_levels - 1)
        pos_quantized = torch.round((pos_weights - min_val) / step) * step + min_val
        pos_quantized = pos_quantized.clamp(min=min_val, max=max_val)
    else:
        pos_quantized = torch.empty_like(pos_weights)
    
    # Quantize negative weights
    neg_weights = weights[weights < 0]
    if len(neg_weights) > 0:
        max_val = neg_weights.max()
        min_val = neg_weights.min()
        step = (max_val - min_val) / (num_levels - 1)
        neg_quantized = torch.round((neg_weights - min_val) / step) * step + min_val
        neg_quantized = neg_quantized.clamp(max=max_val, min=min_val)
    else:
        neg_quantized = torch.empty_like(neg_weights)    
    quantized_weights = torch.zeros_like(weights)
    quantized_weights[weights > 0] = pos_quantized
    quantized_weights[weights < 0] = neg_quantized

    return quantized_weights.detach()

class PoissonEncoder(nn.Module):
    '''
        Implementation of the Poisson encoder that generates poisson spike
        Input of the poissonEncoder should be in the range or [0,1], where input a in [0,1] will have 100*a% of firing a spike.

        :param timesteps: numbers of timesteps that will be generated for a single input 
        :type timesteps: int

        :param scale: reset value whe n the neuron fires
        :type scale: float
        
    '''
    def __init__(self, timesteps:int = 100, scale:float = 1.0):
        super().__init__()
        self.timesteps = timesteps
        self.total_spikes = 0
        self.scale = scale
    def forward(self, x):
        # Convert input to firing rates
        firing_rates = x #the input is resized to [0,1] already
        batch_size, n_pixels = firing_rates.shape
        T_firing_rates = self.scale*firing_rates.unsqueeze(0).repeat(self.timesteps, 1, 1) #[T, B, Pixels]
        # Generate Poisson spikes
        spikes = torch.where(torch.rand_like(T_firing_rates) < T_firing_rates, 1.0, 0.0)
        self.total_spikes+=spikes.sum()
        return spikes
    def get_total_spikes(self):
        return self.total_spikes

class IF_neuron(nn.Module):
    '''
    Implementation of the one step Integrate and fire (IF) Neuron. 
    
    :param v_threshold: threshold to fire the neuron
    :type v_threshold: float

    :param v_reset: reset value when the neuron fires
    :type v_threshold: float
    
    '''
    #When is the neuron reset?
    def __init__(self, v_threshold:float = 1, v_reset:float = 0):
        super().__init__()
        self.v = v_reset
        self.v_threshold = v_threshold
        self.v_threshold_norm = -float("Inf") # spike norm
        self.v_reset = v_reset
    
    def get_total_spikes(self):
        return self.encoder.get_total_spikes()
    def fire(self, input):
        """compute each value to see if the spike reset or not"""
        diff = self.v - self.v_threshold 
        return torch.where(diff> 0, 1.0, 0.0)
    def reset(self, spike:torch):
        """
        for the spiked neuron, 
        reset it to v_reset given the 
        """
        self.v = self.v - spike*self.v_threshold
        
    def reinitialize(self):
        """
        Reinitialized the memristor value back to the intial state
        """
        self.v = self.v_reset
    def spike_normalize(self):
        self.v_threshold = self.v_threshold_norm
    def forward(self, input: Tensor) -> Tensor:
        #init the self.v 
        if isinstance(self.v, float): 
            self.v = torch.full_like(input, self.v_reset)
            print(f'initiating self.v')

        #charge
        self.v = self.v + input
        #spike
        spike = self.fire(input)
        #reset if spike>threshold
        self.reset(spike)
        self.v_threshold_norm = max(self.v_threshold_norm, input.max()) # input= A batch of synapes_input *synapes_weight
        return spike

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}'


class SNN(nn.Module):
    def __init__(self, timesteps=32, firing_scale=1.0, do_spike_norm=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.timesteps = timesteps
        self.encoder = PoissonEncoder(timesteps=self.timesteps, scale=firing_scale)
        self.FCNN = nn.Sequential(
            nn.Linear(28*28, 100, bias=False),
            IF_neuron(),
            nn.Linear(100, 10, bias=False)
        )
        self.do_spike_norm = do_spike_norm
        
        # self.layer1 = nn.Linear(28*28, 100, bias=False)
        # self.IF = IF_neuron()
        # self.layer2 =nn.Linear(100, 10, bias=False)
    def apply_weight_constraints(self):
        #apply weight constraints to 
        self.FCNN[0].weight.data = quantize_weights(self.FCNN[0].weight.data, num_bits=4)

        on_off_ratio = 10
        max_val = self.FCNN[0].weight.data.abs().max()
        min_val = max_val / on_off_ratio
        
        # Clip weights based on ON-OFF ratio
        self.FCNN[0].weight.data[self.FCNN[0].weight.data > 0] = self.FCNN[0].weight.data[self.FCNN[0].weight.data > 0].clamp(min=min_val, max=max_val)

        self.FCNN[0].weight.data[self.FCNN[0].weight.data < 0] = self.FCNN[0].weight.data[self.FCNN[0].weight.data < 0].clamp(min=min_val, max=max_val)
        
        self.FCNN[0].weight.data = quantize_weights(self.FCNN[0].weight.data, num_bits=4)
    
    def forward(self, x):
        net_reinitialize(self) #reinitialize the IF-neuron valt
        x = self.flatten(x)
        x = self.encoder(x)

        logits =  torch.zeros(x.shape[1],10).to(x)
        for t in range(self.timesteps):
            logits += self.FCNN(x[t,:])
        if self.do_spike_norm:
            self.FCNN[1].spike_normalize()
        logits = logits
        
        return logits
#Homeostasis

#Add spikes


class ANN(nn.Module):
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

