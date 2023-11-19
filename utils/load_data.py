import pickle
import matplotlib.pyplot as plt
import numpy as np
import utils.data_process as data_process
from utils.pics import plot_mean

with open('data/ordinary/DQN/ordinary.pkl','rb') as f:
    dict=pickle.load(f)
    f.close()

rewards=dict['rewards']
crashes=dict['crashes']

mean_reward,rhw_reward,var_reward=data_process.calculate(rewards)
mean_crash,rhw_crash,var_crash=data_process.calculate(crashes)

n=len(rewards)
plot_mean(mean_reward,'Number of Episodes','Reward',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_mean.png')
plot_mean(rhw_reward,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_rhw.png')
plot_mean(var_reward,'Number of Episodes','Variance',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_var.png')

plot_mean(mean_crash,'Number of Episodes','Crash Rate',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_mean.png')
plot_mean(rhw_crash,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_rhw.png')
plot_mean(var_crash,'Number of Episodes','Variance',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_var.png')



