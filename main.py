import random
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def set_seed(n_seed):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

# def preprocess(images):
#     gray_images = images[:,:,:,0] * 299/1000 + images[:,:,:,1] * 587/1000 + images[:,:,:,2] * 114/1000
#     return gray_images

def plot_state(s, env_name):
    plt.figure()
    plt.imshow(s)
    plt.savefig('s_'+env_name+'.jpg')
    plt.close()

def plot_obsevation(o, env_name):
    plt.figure()
    plt.imshow(o.view(84,84).numpy(), cmap='gray')
    plt.savefig('o_'+env_name+'.jpg')
    plt.close()

# Experience = namedtuple('Experience', ['observation','action','reward','next_observation','done'])

grayscale_downsample = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((110,84), interpolation=1),
    transforms.ToTensor()
])

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def crop(images):
    return images[:,17:-9,:]

def preprocess(images):
    return (255*crop(grayscale_downsample(images))).byte()

def output_width(input_width, kernel_size, stride):
    return int((input_width - kernel_size)/stride + 1)

class ReplayBuffer():
    def __init__(self, 
                capacity = 50000,
                seed=0):
        self.capacity = capacity
        self.experiences = []        
        self.pointer = 0
    
    def store(self, experience):
        if len(self.experiences) < self.capacity:
            self.experiences.append(None)
        self.experiences[self.pointer] = experience
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample_transitions(self, batch_size):
        if batch_size < len(self.experiences):
            return random.sample(self.experiences, int(batch_size)) 
        else:
            return random.sample(self.experiences, len(self.experiences))

    def retrieve(self):
        return np.copy(self.experiences)

class QNetwork(nn.Module):
    def __init__(self,
                n_actions,
                input_width=84,
                channel_sizes=[4,16,32],
                kernel_sizes=[8,4],
                stride_sizes=[4,2],
                fully_connected_sizes=[256],
                lr=2.5e-4):
        super().__init__()

        self.l1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_sizes[0], stride=stride_sizes[0])
        self.l2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_sizes[1], stride=stride_sizes[1])
        output_width_l1 = output_width(input_width, kernel_sizes[0], stride_sizes[0])
        output_width_l2 = output_width(output_width_l1, kernel_sizes[1], stride_sizes[1])
        self.l3 = nn.Linear(channel_sizes[2]*output_width_l2**2, fully_connected_sizes[0])
        self.l4 = nn.Linear(fully_connected_sizes[0], n_actions)

        self.apply(weights_init_)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        x = F.relu(self.l1(observation))
        x = F.relu(self.l2(x))
        x = x.view(observation.size(0),-1)
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return x

class EpsilonGreedyPolicy:
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
    def __init__(self, 
                beta=1):
        self.beta = beta

    def sample(self, q_values):
        logits = self.beta*q_values - torch.logsumexp(self.beta*q_values)
        return Categorical(logits=logits).sample().item()

class Agent:
    def __init__(self,
                n_actions, 
                buffer_size=1000000,
                behaviour_policy='epsilon_greedy',
                discount_factor=0.99,
                clip_grad_norm_value=10.0,
                policy_args={}
                ):
        self.discount_factor = discount_factor
        self.clip_grad_norm_value = clip_grad_norm_value
        self.replay_buffer = ReplayBuffer(capacity = buffer_size)
        if behaviour_policy == 'epsilon_greedy':
            self.policy = EpsilonGreedyPolicy(policy_args)
        else:
            self.policy = SoftPolicy()
        self.q_network = QNetwork(n_actions).to(device)        

    def remember(self, experience):
        self.replay_buffer.store(experience)
    
    def act(self, observation):
        with torch.no_grad():
            q_values = self.q_network(observation.float().unsqueeze(0).to(device))
            action = self.policy.sample(q_values.cpu())
        return action
    
    def learn(self, batch_size=32):
        transition_batch = self.replay_buffer.sample_transitions(batch_size)
        transition_batch = list(map(list, zip(*transition_batch)))
        observations = torch.stack(transition_batch[0]).float().to(device)
        actions = np.array(transition_batch[1])
        rewards = torch.FloatTensor(transition_batch[2]).view(-1,1).to(device)
        next_observations = torch.stack(transition_batch[3]).float().to(device)
        dones = torch.FloatTensor(transition_batch[4]).view(-1,1).to(device)

        q_values = (self.q_network(observations)[np.arange(observations.size(0)), actions]).view(-1,1)
        next_q_values = (self.q_network(next_observations).max(1)[0]).view(-1,1)
        targets = rewards  +  self.discount_factor * (1-dones) * next_q_values

        q_loss = self.q_network.loss_func(q_values, targets.detach())
        self.q_network.optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(self.q_network.parameters(), self.clip_grad_norm_value)
        self.q_network.optimizer.step()

        return q_loss
        
# env_names = [
#     'BeamRider-v0',
#     'Breakout-v0',
#     'Enduro-v0',
#     'Pong-v0',
#     'Qbert-v0',
#     'Seaquest-v0',
#     'SpaceInvaders-v0'
# ]

env_names = [
            'Pong-v0'
            ]

buffer_size = 4
M = 5
frames_to_skip = 3
max_steps = 50000
step = 0
seed = 1000
history_buffer = []
representation_buffer = []
set_seed(seed)

def initialize_history(state):
    preprocessed_state = preprocess(state)
    for _ in range(0, buffer_size):
        history_buffer.append(preprocessed_state.clone())
    observation = torch.cat(history_buffer)
    return observation

def write_history(state):
    preprocessed_state = preprocess(state)
    history_buffer.append(preprocessed_state.clone())
    while len(history_buffer) > buffer_size:
        history_buffer.pop(0)  
    observation = torch.cat(history_buffer)
    return observation     

for env_name in env_names:
    env = gym.make(env_name)
    n_actions = env.action_space.n
    agent = Agent(n_actions)       
    losses = []

    for episode in range(0, M):
        state = env.reset()
        observation = initialize_history(state)

        frames_skipped = 0
        action = 0
        while step < max_steps:
            if frames_skipped % (frames_to_skip + 1) == 0:
                action = agent.act(observation)
            next_state, reward, done, info = env.step(action)

            if frames_skipped % (frames_to_skip + 1) == 0:
                next_observation = write_history(next_state)            
                experience = [observation, action, reward, next_observation, done]
                agent.remember(experience)
                loss = agent.learn()            
                losses.append(loss)

                stdout.write("Epsd %i, step %i, loss: %.4f" % (episode+1, step+1, loss))

                step += 1

            if done:
                history_buffer = []
                break
            
            frames_skipped += 1

    np.savetxt('losses_'+env_name+'.txt', np.array(losses))
