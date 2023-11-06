import gymnasium as gym
import gym_testenvs
import utils.data_process as data_process
from utils.pics import plot_mean

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt

import pickle
import tqdm

n=100000

env = gym.make('LunarLander/ordinary-v0',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl")

# def reward():
#     episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=n,return_episode_rewards = True,render=False)

#     mean,rhw,var=data_process.calculate(episode_rewards)
        
#     plot_mean(mean,'Number of Episodes','Reward',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_mean.png')
#     plot_mean(rhw,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_rhw.png')
#     plot_mean(var,'Number of Episodes','Variance',['Ordinary method'],[0,1000],save_path='data/reward/ordinary_var.png')
rewards = []
crashes = []
for i in tqdm.tqdm(range(n)):
    obs,_= env.reset()
    done = False
    reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward_, tr, crash, info = env.step(action)
        done= crash or tr
        reward+=reward_
        if done:
            rewards.append(reward)
            crashes.append(int(crash))

print(crashes)
dict={'rewards':rewards,'crashes':crashes}
with open('data/ordinary/ordinary.pkl','wb') as f:
    pickle.dump(dict,f)
    f.close()

mean_reward,rhw_reward,var_reward=data_process.calculate(rewards)
mean_crash,rhw_crash,var_crash=data_process.calculate(crashes)

print(mean_crash)

plot_mean(mean_reward,'Number of Episodes','Reward',['Ordinary method'],[0,100000],save_path='data/ordinary/reward/ordinary_mean.png')
plot_mean(rhw_reward,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,100000],save_path='data/ordinary/reward/ordinary_rhw.png')
plot_mean(var_reward,'Number of Episodes','Variance',['Ordinary method'],[0,100000],save_path='data/ordinary/reward/ordinary_var.png')

plot_mean(mean_crash,'Number of Episodes','Crash Rate',['Ordinary method'],[0,100000],save_path='data/ordinary/crash/ordinary_mean.png')
plot_mean(rhw_crash,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,100000],save_path='data/ordinary/crash/ordinary_rhw.png')
plot_mean(var_crash,'Number of Episodes','Variance',['Ordinary method'],[0,100000],save_path='data/ordinary/crash/ordinary_var.png')


    