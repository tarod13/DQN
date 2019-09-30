import numpy as np
import torch
import gym
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import namedtuple

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


Experience = namedtuple('Experience', ['observation','action','reward','next_observation','done'])

grayscale_downsample = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((110,84), interpolation=1),
    transforms.ToTensor()
])

def crop(images):
    return images[:,17:-9,:]

def preprocess(images):
    return (255*crop(grayscale_downsample(images))).byte()

env_names = [
    'BeamRider-v0',
    'Breakout-v0',
    'Enduro-v0',
    'Pong-v0',
    'Qbert-v0',
    'Seaquest-v0',
    'SpaceInvaders-v0'
]

buffer_size = 4
M = 5
max_steps = 50000
step = 0
history_buffer = []
representation_buffer = []

def initialize_history(state):
    preprocessed_state = preprocess(state)
    for _ in range(0, buffer_size):
        history_buffer.append(preprocessed_state.clone())
    observation = torch.cat(history_buffer)
    return observation

def write_history(state):
    preprocessed_state = preprocess(state)
    history_buffer.append(preprocessed_state.clone())
    while len(history_buffer) >= buffer_size:
        history_buffer.pop(0)  
    observation = torch.cat(history_buffer)
    return observation     

for env_name in env_names:
    env = gym.make(env_name)       

    for episode in range(0, M):
        state = env.reset()
        observation = initialize_history(state)

        while step < max_steps:
            action = agent.act(observation)
            next_state, reward, done, info = env.step(action)
            next_observation = write_history(next_state)
            
            experience = Experience(observation, action, reward, next_observation, done)
            agent.remember(experience)
            agent.learn()            
            
            step += 1

            if done:
                history_buffer = []
                break