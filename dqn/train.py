import numpy as np
from sys_and_agent import System
import os

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 100
seed = 1000
last_iter = 0
tr_epsds = 2000
epsd_steps = 1000
n_iter = 1
env_name = 'Pong-v0'
basic_epsds = 0
n_basic_tasks = 0

folder_name = 'Pong'
common_path = folder_name + '/' + str(n_test)
if not exists_folder(folder_name):
    os.mkdir(folder_name)
if not exists_folder(common_path):
    os.mkdir(common_path)

system = System(env_name=env_name)
started = False
if last_iter > 0:
    try:
        specific_path = common_path + '/' + str(last_iter)
        system.load(common_path, specific_path, load_upper_memory=False)
        mean_rewards = list(np.loadtxt(common_path + '/mean_rewards.txt'))
        started = True
    except:
        pass         
if not started:
    last_iter = 0
    mean_rewards = []    

system.train_agent( tr_epsds, epsd_steps, iter_=last_iter, rewards=mean_rewards, common_path=common_path)