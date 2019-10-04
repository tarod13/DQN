import numpy as np
import torch
from torch.distributions import Categorical

class EpsilonGreedyPolicy:
    '''
    Policy that selects the action that maximizes the q-value with probability
    1-epsilon and samples from a uniform distribution with probability epsilon.
    '''
    def __init__(self,
                annealing=True,
                min_value=0.1,
                initial_value=1.0,
                **kwargs):
        self.epsilon = initial_value
        self.min_value = min_value
        self.annealing = annealing
        if self.annealing:
            if 'steps' in kwargs:
                steps = kwargs['steps']
            else:
                steps = 1000000
            self.delta = (initial_value - min_value)/steps

    def sample(self, q_values, update_epsilon=True):
        if self.epsilon < np.random.rand():
            action = np.random.randint(len(q_values)) 
        else:
            action = q_values.argmax().item()

        if self.annealing and update_epsilon:
            self.epsilon = np.min([self.epsilon-self.delta,self.min_value])

        return action


class SoftPolicy:
    '''
    Policy that choses actions based on a soft-max distribution of the q-values,
    with temperature parameter 1/beta.
    '''
    def __init__(self, 
                beta=1):
        self.beta = beta

    def sample(self, q_values):
        logits = self.beta*q_values - torch.logsumexp(self.beta*q_values)
        return Categorical(logits=logits).sample().item()
