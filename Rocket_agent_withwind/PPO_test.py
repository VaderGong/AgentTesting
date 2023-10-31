import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('LunarLander-v2',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0,render_mode='human')#render_mode='human'可视化
model = PPO.load("model/ppo_lunar.pkl")

def mean_value_test():
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000,render=False)#render=True慢速render
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_value_test()