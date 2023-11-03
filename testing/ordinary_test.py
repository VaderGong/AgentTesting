import gymnasium as gym
import gym_testenvs
import data_process

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt

n=1000

env = gym.make('LunarLander/ordinary-v0',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

model = DQN.load("../Rocket_agent_withwind/model/dqn_lunar.pkl")

episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=n,return_episode_rewards = True,render=False)

mean,rhw,var=data_process.calculate(episode_rewards)

print(mean)
print(rhw)
print(var)

def plot_mean(ys,xlabel,ylabel,legend,xlim,save_path=None):
    plt.plot(ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_rhw(relative_half_width):
    plt.plot(relative_half_width)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Relative Half Width')
    plt.legend(['Ordinary method'])
    plt.show()
    
plot_mean(mean)
