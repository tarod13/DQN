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

from nets import QNetwork
from policies import EpsilonGreedyPolicy, SoftPolicy
from buffers import ReplayBuffer

import os
import pickle
from sys import stdout
import itertools

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image

width = 1024
height = 768
FPS = 60

fourcc = VideoWriter_fourcc(*'MP42')

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def set_seed(n_seed):
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

#-------------------------------------------------------------
#
#                           Classes
#
#-------------------------------------------------------------
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
    
    def save(self, common_path, specific_path):
        pickle.dump(self.memory,open(common_path+'/memory.p','wb'))
        pickle.dump(self.policy,open(specific_path+'_policy.p','wb'))
        torch.save(self.q_network.state_dict(), specific_path+'_q_network.pt')
        
    def load(self, common_path, specific_path, load_upper_memory=True):
        self.memory = pickle.load(open(common_path+'/memory.p','rb'))
        self.policy = pickle.load(open(specific_path+'_policy.p','rb'))
        self.q_network.load_state_dict(torch.load(specific_path+'_q_network.pt'))
        
        self.q_network.eval()

            
class System:
    def __init__(self, 
            seed=1000,
            env_name='Pong-v0',
            buffer_size=500000,
            behaviour_policy='epsilon_greedy',
            clip_grad_norm_value=10.0,
            policy_args={}, 
            env_steps=1, 
            grad_steps=1, 
            init_steps=10000, 
            discount_factor=0.99,
            beta=1.0, 
            batch_size=32, 
            q_lr=3e-4, 
            render=True, 
            reset_when_done=True, 
            store_video=False, 
            annealing=False, 
            max_eta=8e-2):

        set_seed(seed)
        self.seed = seed
        self.env_name = env_name
        self.set_env()
        
        self.env_steps = env_steps
        self.grad_steps = grad_steps
        self.init_steps = init_steps
        self.batch_size = batch_size
        self.render = render
        self.store_video = store_video
        self.reset_when_done = reset_when_done
        
        self.buffer_size = 4
        self.frames_to_skip = 3
        self.max_steps = 50000
        self.step = 0
        self.seed = 1000
        self.history_buffer = []

        self.grayscale_downsample = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Grayscale(num_output_channels=1),
                                                    transforms.Resize((110,84), interpolation=1),
                                                    transforms.ToTensor()])

        n_actions = self.env.action_space.n
        self.agent = Agent(n_actions, 
                        buffer_size=buffer_size,
                        behaviour_policy=behaviour_policy,
                        discount_factor=discount_factor,
                        clip_grad_norm_value=clip_grad_norm_value,
                        policy_args=policy_args)        

    def set_env(self):
        self.env = gym.make(self.env_name)
        print("Created env "+self.env_name)
        self.env.reset()
        self.env.seed(self.seed)        
        self.env.rgb_rendering_tracking = True
    
    def crop(self, images):
        return images[:,17:-9,:]

    def preprocess(self, images):
        return (255*self.crop(self.grayscale_downsample(images))).byte()
    
    def initialize_history(self, state):
        preprocessed_state = self.preprocess(state)
        for _ in range(0, self.buffer_size):
            self.history_buffer.append(preprocessed_state.clone())
        observation = torch.cat(self.history_buffer)
        return observation

    def write_history(self, state):
        preprocessed_state = self.preprocess(state)
        self.history_buffer.append(preprocessed_state.clone())
        while len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)  
        observation = torch.cat(self.history_buffer)
        return observation

    def reset(self, change_env=False):        
        self.env.reset()
       
    def train_agent(self, tr_epsds, epsd_steps, eval_epsd_interval=10, eval_epsds=12, iter_=0, save_progress=True, common_path='', rewards=[], metrics=[]):        
        if self.render:
            self.env.render()

        losses = []
        rewards = []
        for epsd in range(0, tr_epsds):
            state = self.env.reset()
            observation = self.initialize_history(state)
            frames_skipped = 0
            action = 0
            epsd_losses = []

            for epsd_step in range(0, epsd_steps):
                if frames_skipped % (self.frames_to_skip + 1) == 0:
                    action = self.agent.act(observation)
                next_state, reward, done, info = self.env.step(action)

                if self.render:
                    self.env.render()

                if frames_skipped % (self.frames_to_skip + 1) == 0:
                    next_observation = self.write_history(next_state)            
                    experience = [observation, action, reward, next_observation, done]
                    self.agent.remember(experience)
                    loss = self.agent.learn()            
                    epsd_losses.append(loss)

                    stdout.write("Epsd %i, step %i, loss: %.4f" % (epsd+1, epsd_step+1, loss))

                if done:
                    self.history_buffer = []
                    break
                
                frames_skipped += 1

            losses.append(epsd_losses)
            if save_progress:
                pickle.dump(losses, open(common_path+'/losses.p','wb'))              
                
            if (epsd+1) % eval_epsd_interval == 0:
                rewards.append(self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[0])
                if save_progress:
                    specific_path = common_path + '/' + str(iter_ + (epsd+1) // eval_epsd_interval)
                    self.save(common_path, specific_path)
                    np.savetxt(common_path + '/mean_rewards.txt', np.array(rewards))
              
        return np.array(rewards).reshape(-1), losses      
    
    def eval_agent(self, eval_epsds, act_randomly=False, iter_=0, start_render=False, print_space=True):   
        if start_render:
            self.env.render()
          
        if self.store_video:
            video = VideoWriter('./'+self.env_name+'_test_'+str(iter_)+'.avi', fourcc, float(FPS), (width, height))          
        
        events = []
        rewards = []
        min_epsd_reward = 1e6
        max_epsd_reward = -1e6
        done = False
        
        for epsd in range(0, eval_epsds):
            epsd_reward = 0.0
            
            self.reset(change_env=not done)
            
            for eval_step in itertools.count(0):            
                event = self.interaction(learn=False, explore=act_randomly, remember=False, epsd_step=eval_step)[0]                

                if self.store_video:
                    if self.env_names[self.task] == 'Pendulum-v0':
                        img = self.envs[self.task].render('rgb_array')
                    else:
                        img = self.envs[self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if self.render:
                    self.envs[self.task].render()   

                r = event[self.sa_dim]
                if self.n_tasks > 1 or self.embedded_envs:
                    R = event[self.sarsd_dim]
                done = event[self.sars_dim]

                epsd_reward += r
                if self.n_tasks > 1 or self.embedded_envs:
                    epsd_goal_reward += R
                events.append(event)

                if done or (eval_step + 1 >= self.envs[self.task]._max_episode_steps):
                    break               

            if self.hierarchical:
                metrics = self.agent.learn(only_metrics=True)
                uniques.append(metrics['n concepts'])
                HS.append(metrics['H(S)'])
                HS_s.append(metrics['H(S|s)'])
                ISs.append(metrics['I(S:s)'])
                Ha_s.append(metrics['H(a|s)'])
                Ha_As.append(metrics['H(a|A,s)'])
                HnS_AT.append(metrics['H(nS|A,T)'])
                HnS_SAT.append(metrics['H(nS|S,A,T)'])
                InSS_AT.append(metrics['I(nS:S|A,T)'])
                HR_ST.append(metrics['H(R|S,T)'])
                HR_T.append(metrics['H(R|T)'])
                IRS_T.append(metrics['I(R:S|T)'])
           
            rewards.append(epsd_reward)
            min_epsd_reward = np.min([epsd_reward, min_epsd_reward])
            max_epsd_reward = np.max([epsd_reward, max_epsd_reward])
            average_reward = np.array(rewards).mean()  

            if self.n_tasks > 1 or self.embedded_envs:
                goal_rewards.append(epsd_goal_reward)            
                min_epsd_goal_reward = np.min([epsd_goal_reward, min_epsd_goal_reward])
                max_epsd_goal_reward = np.max([epsd_goal_reward, max_epsd_goal_reward])           
                average_goal_reward = np.array(goal_rewards).mean()

            if self.hierarchical:
                unique_average += (uniques[-1] - unique_average)/(epsd+1)
                ISs_average += (ISs[-1] - ISs_average)/(epsd+1)
                HR_ST_average += (HR_ST[-1] - HR_ST_average)/(epsd+1)
                HR_T_average += (HR_T[-1] - HR_T_average)/(epsd+1)
                IRS_T_average += (IRS_T[-1] - IRS_T_average)/(epsd+1)
                InSS_AT_average += (InSS_AT[-1] - InSS_AT_average)/(epsd+1)
                HS_average += (HS[-1] - HS_average)/(epsd+1)
                HS_s_average += (HS_s[-1] - HS_s_average)/(epsd+1)
                Ha_s_average += (Ha_s[-1] - Ha_s_average)/(epsd+1)
                Ha_As_average += (Ha_As[-1] - Ha_As_average)/(epsd+1)
                HnS_AT_average += (HnS_AT[-1] - HnS_AT_average)/(epsd+1)
                HnS_SAT_average += (HnS_SAT[-1] - HnS_SAT_average)/(epsd+1)

            if self.hierarchical and (self.n_tasks > 1 or self.embedded_envs):
                stdout.write("Iter %i, epsd %i, u: %.2f, I(s:S): %.4f, I(r:S):%.3f, I(nS:S): %.4f, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f, min R: %.1f, max R: %.1f, mean R: %.2f, epsd R: %.1f\r " %
                    (iter_, (epsd+1), unique_average, ISs_average, IRS_T_average, InSS_AT_average, min_epsd_reward, max_epsd_reward, average_reward, epsd_reward, 
                        min_epsd_goal_reward, max_epsd_goal_reward, average_goal_reward, epsd_goal_reward))
                stdout.flush() 
            elif self.hierarchical and not (self.n_tasks > 1 or self.embedded_envs):
                stdout.write("Iter %i, epsd %i, u: %.2f, I(s:S): %.4f, I(r:S):%.3f, I(nS:S): %.4f, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f\r " %
                    (iter_, (epsd+1), unique_average, ISs_average, IRS_T_average, InSS_AT_average, min_epsd_reward, max_epsd_reward, average_reward, epsd_reward))
                stdout.flush()
            elif not self.hierarchical and (self.n_tasks > 1 or self.embedded_envs):
                stdout.write("Iter %i, epsd %i, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f, min R: %.1f, max R: %.1f, mean R: %.2f, epsd R: %.1f\r " %
                    (iter_, (epsd+1), min_epsd_reward, max_epsd_reward, average_reward, epsd_reward, min_epsd_goal_reward, max_epsd_goal_reward, average_goal_reward, epsd_goal_reward))
                stdout.flush()
            else:
                stdout.write("Iter %i, epsd %i, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f\r " %
                    (iter_, (epsd+1), min_epsd_reward, max_epsd_reward, average_reward, epsd_reward))
                stdout.flush()            

        if print_space:    
            print("")
            
        if self.store_video:
            video.release()
        # if self.render:
        #     self.envs[self.task].close()   

        if self.hierarchical:
            metric_vector = np.array([Ha_As_average, HS_average, HS_s_average, ISs_average, unique_average, HnS_AT_average, HnS_SAT_average, InSS_AT_average, HR_ST_average, HR_T_average, IRS_T_average, Ha_s_average, Ha_s_average-Ha_As_average]) 
            if self.n_tasks > 1 or self.embedded_envs:
                return rewards, goal_rewards, np.array(events), metric_vector
            else:
                return rewards, np.array(events), metric_vector
        else:
            if self.n_tasks > 1 or self.embedded_envs:
                return rewards, goal_rewards, np.array(events)
            else:
                return rewards, np.array(events)
    
    def save(self, common_path, specific_path):
        self.agent.save(common_path, specific_path)
    
    def load(self, common_path, specific_path, load_upper_memory=True):
        self.agent.load(common_path, specific_path, load_upper_memory=load_upper_memory)