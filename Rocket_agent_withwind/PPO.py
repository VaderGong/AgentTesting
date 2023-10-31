import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import torch
env = gym.make('LunarLander-v2',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)
model = PPO(policy='MlpPolicy',
            env=env,
            n_steps=1024,
            batch_size=64,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4,
            ent_coef=0.01
            )
logger=configure("logger/PPO",["tensorboard","stdout","csv"])
model.set_logger(logger)
model.learn(total_timesteps=1.1e6, log_interval=4, progress_bar=True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
model.save("model/ppo_lunar.pkl")