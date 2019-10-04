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
        self.epsd_counter = 0

        self.buffer_size = 4
        self.M = 5
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
        if self.embedded_envs:
            if self.env_names[self.task] == 'Pendulum-v0' and self.hard_start:
                self.envs[self.task].state = np.array([-np.pi,0.0])
            else:
                self.envs[self.task].reset()
                self.task = self.envs[0]._task
        else:
            if change_env:
                self.task = (self.task+1) % self.task_modulo
            if self.env_names[self.task] == 'Pendulum-v0' and self.hard_start:
                self.envs[self.task].state = np.array([-np.pi,0.0])
            else:
                self.envs[self.task].reset()
            self.agent.reset_upper_level()
    
    def get_obs(self):
        if self.original_state:
            state = self.envs[self.task]._get_obs().copy()
        else:
            if self.env_names[self.task] == 'Pendulum-v0':            
                state = self.envs[self.task].state.copy().reshape(-1) 
                state[0] = normalize_angle(state[0])
            elif self.env_names[self.task] == 'Ant-v3':
                state = self.envs[self.task]._get_obs()[:28]
        return state

    def initialization(self):         
        self.reset()
        self.epsd_counter += 1
        average_r = 0.0
        epsd_step = 0
        for init_step in range(0, self.init_steps):
            epsd_step += 1           
            event = self.interaction_init(epsd_step)
            r = event[self.sa_dim]
            done = event[self.sars_dim]
            average_r += (r-average_r)/(init_step+1)
            if done:
                epsd_step = 0
            if self.render:
                self.envs[self.task].render()                        
        print("Finished initialization, av. reward = %.4f" % (average_r))

    def interaction_init(self, epsd_step):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action, action_llhood = self.agent.act(state, self.task, explore=True)
        scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
        _, reward, done, info = self.envs[self.task].step(scaled_action)  
        done = done and self.reset_when_done
        next_state = self.get_obs()
        if done:
            self.reset(change_env=True)                   
        
        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim+1] = self.task
        
        if self.n_tasks > 1 or self.embedded_envs:
            event[self.sarsd_dim] = info['reward_goal']  
        else:
            event[self.sarsd_dim] = reward
        self.agent.collect_reward(event[self.sarsd_dim], done, self.task, epsd_step>=self.envs[self.task]._max_episode_steps, state, action, action_llhood)    
        
        self.agent.memorize(event)   
        return event

    def interaction(self, learn=True, remember=True, init=False, explore=True, epsd_step=0):  
        event = np.empty(self.t_dim)
        state = self.get_obs()

        for env_step in range(0, self.env_steps):
            action, action_llhood = self.agent.act(state, self.task, explore=explore)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            _, reward, done, info = self.envs[self.task].step(scaled_action)
            # if (epsd_step*self.env_steps+1) >= 1000:
            #     print("Done "+str(done))
            done = done and self.reset_when_done # must be changed if done == True when time == max_time
            next_state = self.get_obs()                            

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            event[self.sarsd_dim+1] = self.task
        
            if self.n_tasks > 1 or self.embedded_envs:
                event[self.sarsd_dim] = info['reward_goal']                  
            else:
                event[self.sarsd_dim] = reward    
            
            if remember:
                self.agent.memorize(event)

            if remember and (self.hierarchical or self.n_tasks > 1 or self.embedded_envs):
                self.agent.collect_reward(event[self.sarsd_dim], done, self.task, (epsd_step*self.env_steps+env_step+1)>=self.envs[self.task]._max_episode_steps, state, action, action_llhood)

            if done:
                self.reset(change_env=True)
                break

            if env_step < self.env_steps-1:
                state = np.copy(next_state)
        
        if learn and not init:
            for _ in range(0, self.grad_steps):
                self.agent.learn()

        return event, done
    
    def train_agent(self, tr_epsds, epsd_steps, initialization=True, eval_epsd_interval=10, eval_epsds=12, iter_=0, save_progress=True, common_path='', rewards=[], goal_rewards=[], metrics=[]):        
        if self.render:
            self.envs[self.task].render()

        if initialization:
            self.initialization()

        n_done = 0
        # rewards = []
        # if self.n_tasks > 1 or self.embedded_envs:
        #     goal_rewards = []

        for epsd in range(0, tr_epsds):
            self.epsd_counter += 1
            if epsd == 0:
                self.reset(change_env=False)
            else:
                self.reset(change_env=True)
            
            for epsd_step in range(0, epsd_steps):
                if len(self.agent.memory.data) < self.batch_size:
                    done = self.interaction(learn=False, epsd_step=epsd_step)[1]
                else:
                    done = self.interaction(learn=True, epsd_step=epsd_step)[1]

                if self.render:
                    self.envs[self.task].render()

                if done:
                    n_done += 1
                
                if n_done >= 5:
                    self.eval_agent(1, act_randomly=False, iter_=iter_, print_space=False)
                    self.reset(change_env=False)
                    n_done = 0
            
            if (epsd+1) % eval_epsd_interval == 0:
                if self.hierarchical:
                    if self.n_tasks > 1 or self.embedded_envs:
                        r, gr, _, m = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)
                        goal_rewards.append(gr)
                        if save_progress:
                            np.savetxt(common_path + '/mean_rewards_goal.txt', np.array(goal_rewards))
                    else:
                        r, _, m = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)
                    metrics.append(m)
                    if save_progress:
                        np.savetxt(common_path + '/metrics.txt', np.array(metrics))
                else:
                    if self.n_tasks > 1 or self.embedded_envs:
                        r, gr = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[:2]
                        goal_rewards.append(gr)
                        if save_progress:
                            np.savetxt(common_path + '/mean_rewards_goal.txt', np.array(goal_rewards))
                    else:
                        rewards.append(self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[0])
                rewards.append(r)
                self.reset(change_env=False)
                if save_progress:
                    specific_path = common_path + '/' + str(iter_ + (epsd+1) // eval_epsd_interval)
                    self.save(common_path, specific_path)
                    np.savetxt(common_path + '/mean_rewards.txt', np.array(rewards))
              
        if self.n_tasks > 1 or self.embedded_envs:
            return np.array(rewards).reshape(-1), np.array(goal_rewards).reshape(-1)
        else:      
            return np.array(rewards).reshape(-1)      
    

    def eval_agent(self, eval_epsds, act_randomly=False, iter_=0, start_render=False, print_space=True):   
        if start_render:
            self.envs[self.task].render()
          
        if self.store_video:
            if self.env_names[self.task] == 'Pendulum-v0':
                video = VideoWriter('./'+self.env_names[self.task]+'_test_'+str(iter_)+'.avi', fourcc, float(FPS), (500, 500))
            else:
                video = VideoWriter('./'+self.env_names[self.task]+'_test_'+str(iter_)+'.avi', fourcc, float(FPS), (width, height))          
        
        events = []
        rewards = []
        min_epsd_reward = 1e6
        max_epsd_reward = -1e6
        
        if self.n_tasks > 1 or self.embedded_envs:
            goal_rewards = [] 
            min_epsd_goal_reward = 1e6
            max_epsd_goal_reward = -1e6
        
        if self.hierarchical:
            uniques = []
            HS = []
            HS_s = []
            ISs = []
            Ha_s = []
            Ha_As = []
            HnS_AT = []
            HnS_SAT = []
            InSS_AT = []
            HR_ST = []
            HR_T = []
            IRS_T = []
            unique_average = 0
            HS_average = 0
            HS_s_average = 0
            ISs_average = 0
            HR_ST_average = 0
            HR_T_average = 0
            IRS_T_average = 0
            InSS_AT_average = 0
            Ha_s_average = 0
            Ha_As_average = 0
            HnS_AT_average = 0
            HnS_SAT_average = 0

        done = False
        for epsd in range(0, eval_epsds):
            epsd_reward = 0.0
            if self.n_tasks > 1 or self.embedded_envs:
                epsd_goal_reward = 0.0            

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