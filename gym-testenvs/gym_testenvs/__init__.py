from gymnasium.envs.registration import register
 
register(
    id="LunarLander/ordinary-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_ordinary",
    max_episode_steps=1000,
    reward_threshold=200,
)
