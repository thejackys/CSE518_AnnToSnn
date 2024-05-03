# EE 518 Neuromorphic Computing Spring 2024 Final Project
neuromorphic software simulation project for converting Ann to SNN.

This project implements a Spiking Neural Network (SNN) for digit classification on the MNIST dataset. It explores the conversion of an Artificial Neural Network (ANN) to an SNN, and analyzes the impact of various factors such as firing rates, timesteps, and hardware constraints on the performance and energy consumption of the SNN.

Spike norm and surrogate gradient function are tested to see the difference in the IF_neuron. 
## Spike norm
Spike norm is Adapted from spike-norm proposed from <cite>[Going Deeper in Spiking Neural Networks: VGG and Residual Architectures][1]</cite>. The new threshold voltage for each Spiking neuron layer is set to $$v_{th} = \text{max}(v_{th}, w_0^\intercal s_0, (w_1^\intercal s_1), ..., (w_B^\intercal s_B)) $$ where $B$ is the batch size, $s_i$ is the input from the previous layer, and $w_i$ is the weight respective to $s_i$  after a whole timesteps T is has passed. In the original paper, It's effectively updated with batch size of 1.

## Surrogate gradient function
Neuron firing of the IF-neuron follows the heaviside function $$ S[t] \Theta(v_{mem}[t] - v_{th})$$
The gradient would thus be unstable if follow the autograd directly. Thus (Surrogate Gradient Learning in Spiking Neural Networks)[https://arxiv.org/abs/1901.09948] is proposed to use surrogate function when doing backpropagation.  

Although I am still not sure about the bioplausability of it, I tested the surrogate by applying the sigmoid surrogate gradient function from [spikinjelly](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/surrogate.html)


## Project Structure

- `main.py`: The main script that trains the ANN and SNN models, and runs experiments with different configurations.
- `model.py`: Contains the implementation of the ANN and SNN models, including the Poisson encoder, IF neuron, and surrogate gradient.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- wandb (for logging metrics)
- spikingjelly (for surrogate gradient)

## Usage

1. Install the required dependencies.
2. Run `main.py` to train the models and perform experiments.

## Experiments

The project conducts the following experiments:

1. Trained an ANN with 1 hidden layer (100 neurons) on the MNIST dataset for a basic accuracy
For each mode:
1. Converting the ANN to an SNN and analyzing the accuracy and energy consumption (total number of Accumulate operations) for different timesteps and firing rates.
2.  By deciding the optimal value of timesteps and firingrate, further, Incorporating hardware constraints such as resistance ON-OFF ratio of 10 and 16 discrete resistance states into the SNN model.

## Results

| Model                               | Best Timesteps | Average Runtime (seconds) | Best Firing Rate | Accuracy Threshold |
|-------------------------------------|----------------|---------------------------|------------------|-------------------|
| vanilla                             | 32             | 72                        | 0.46             | > 85%             |
| spike norm                          | 128            | 240                       | 0.76             | > 82%             |
| surrogate function                  | 64             | 200                       | 0.24             | > 96%             |
| spike norm + surrogate function     | 8              | 55                        | 0.38             | > 90%             |
<!-- ### vanilla
best timesteps = 32, average runtinme:72 seconds, best firing rate = 0.46 to maintain accuracy > 85%
### spike norm
best timesteps = 128, average runtime = 240 seconds, best firing rate = 0.76 to maintain accuracy > 82%

### surrogate function
best timesteps = 64, average runtinme = 200 seconds, best firing rate = 0.24 to maintain accuracy > 96%

### spkike norm + surrogate function
best timesteps = 8, average runtinme = 55 seconds, best firing rate = 0.38 to maintain accuracy > 90% -->



## Implementation detail
### Neuron_class
The `IF_neuron` class represents an implementation of the Integrate-and-Fire (IF) neuron model, which is a simple spiking neuron model used in Spiking Neural Networks (SNNs). Here's an explanation of the key components and functionality of the `IF_neuron` class:

The `Surrogate_IF_neuron` replaces the firing method to the surrogate gradient function from spikingjelly.

#### Initialization

- The `IF_neuron` class is initialized with two optional parameters:
 - `v_threshold`: The threshold voltage for firing (default: 1).
 - `v_reset`: The reset voltage after firing (default: 0).
- The neuron's membrane potential `self.v` is initially set to the reset voltage `v_reset`.
- The `self.v_threshold_norm` is set to negative infinity, which is used for spike normalization.

#### Forward Pass

- The `forward` method takes an input tensor and updates the neuron's state accordingly.
- If `self.v` is a float (indicating it hasn't been initialized yet), it is converted to a tensor with the same shape as the input and initialized with the reset voltage.
- The input is added to the neuron's membrane potential `self.v`, representing the integration of the input over time.
- The `fire` method is called to determine if the neuron should spike based on its membrane potential.
- If the neuron spikes, the `reset` method is called to reset the membrane potential to the reset voltage.
- The maximum value of the input is used to update `self.v_threshold_norm` for spike normalization.
- The method returns the spike output tensor.

#### Firing Mechanism

- The `fire` method computes the difference between the membrane potential `self.v` and the threshold voltage `self.v_threshold`.
- It returns a tensor where values greater than 0 are set to 1.0 (indicating a spike) and values less than or equal to 0 are set to 0.0 (indicating no spike).

#### Reset Mechanism

- The `reset` method is called when the neuron fires.
- It subtracts the threshold voltage multiplied by the spike tensor from the membrane potential `self.v`, effectively resetting the membrane potential for the spiked neurons.

#### Reinitialization and Spike Normalization

- The `reinitialize` method sets the membrane potential `self.v` back to the reset voltage `v_reset`, allowing the neuron to be reused for multiple samples.
- The `spike_normalize` method sets the threshold voltage `self.v_threshold` to the value of `self.v_threshold_norm`, which is used for spike normalization.

## Hardware constraints
The Hardware constraints are applied between training and test process:
``` python
train(train_dataloader, model, loss_func, optimizer)
if add_constraint:
    model.apply_weight_constraints()
test(test_dataloader, model, loss_func)
```
To optimize after applying the constraint weight. I use the similar procedure in [Deep Compression]([2]). 
Retrain and apply the constraints recurrsively until a fixed epoch or an desired accuracy is achieved.  
### Resistance ON-Off ratio

### Discrete states
The function `quantize_weight` mimics the discrete state nature in the hardware by setting the positive and negative weights to each have 16 states of weight.    



## Acknowledgements

- The project uses the MNIST dataset and PyTorch framework for deep learning.
- The SNN implementation is based on the concepts and techniques from the field of neuromorphic computing.
- The spikingjelly library is used for implementing the surrogate gradient.


[1]: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00095/full
[2]: https://arxiv.org/abs/1510.00149
