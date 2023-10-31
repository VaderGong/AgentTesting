import gymnasium as gym
import gym_testenvs
import pic

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

n=1000

env = gym.make('LunarLander/ordinary-v0',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

model = DQN.load("../Rocket_agent_withwind/model/dqn_lunar.pkl")

episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=n,return_episode_rewards = True,render=False)

mean,rhw,var=pic.calculate(episode_rewards)

print(mean)
print(rhw)
print(var)
    
