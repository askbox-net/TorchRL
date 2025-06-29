import gymnasium as gym
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env_id = 'Acrobot-v1'
env_model = f'ppo_{env_id}'

# Parallel environments
vec_env = make_vec_env(env_id, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)

if os.path.exists(f'{env_model}.zip'):
    model = PPO.load(env_model, vec_env, verbose=1)

model.learn(total_timesteps=125000)
model.save(env_model)

del model # remove to demonstrate saving and loading

model = PPO.load(env_model)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(0.01)
    if dones[0]:
        obs = vec_env.reset()
        break

