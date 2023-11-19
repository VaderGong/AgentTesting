import gymnasium as gym
import gym_testenvs
import utils.process as process
from utils.pics import plot_mean

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt

import pickle
import tqdm

n=1000000

env = gym.make('LunarLander/ordinary-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device='cpu')

# def reward():
#     episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=n,return_episode_rewards = True,render=False)

#     mean,rhw,var=data_process.calculate(episode_rewards)
        
#     plot_mean(mean,'Number of Episodes','Reward',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_mean.png')
#     plot_mean(rhw,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_rhw.png')
#     plot_mean(var,'Number of Episodes','Variance',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_var.png')

crash_num=256
for i in tqdm.tqdm(range(n)):
    data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'wind_add':[],'torque_add':[],'reward':[],'crash':0}
    obs,_= env.reset()
    done = False
    reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        data_episode['obs'].append(obs)
        data_episode['action'].append(action)
        data_episode['wind_idx'].append(env.wind_idx)
        data_episode['torque_idx'].append(env.torque_idx)
        data_episode['wind_add'].append(env.wind_add)
        data_episode['torque_add'].append(env.torque_add)
        obs, reward_, tr, crash, info = env.step(action)
        data_episode['reward'].append(reward_)
        done= crash or tr
        reward+=reward_
        if crash:
            crash_num+=1
            data_episode['crash']=1
            with open('data/crash/'+str(crash_num)+'.pkl','wb') as f:
                pickle.dump(data_episode,f)
                f.close()

print(crash_num)


