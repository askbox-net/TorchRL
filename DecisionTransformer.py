# https://huggingface.co/hishamcse/decision-transformer-halfcheetah-v4
# https://www.kaggle.com/code/syedjarullahhisham/drl-adv-hf-decisiontransformer-halfcheetah-hopper
from transformers import DecisionTransformerModel
import torch
import gymnasium as gym
import numpy as np


# Function that gets an action from the model using autoregressive prediction with a window of the 
# previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    
    # pad all tokens to sequence length
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

# build the environment
directory = './video'

"""
env = gym.make("Hopper-v4", render_mode='rgb_array')
# env = Recorder(env, directory, fps=30)
env = gym.wrappers.RecordVideo(env, directory)
"""

env = gym.make("Hopper-v4", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

trigger = lambda t: t % 10 == 0 #動画を保存をするエピソードの指定
#env = gym.wrappers.RecordVideo(env, video_folder="./", episode_trigger=trigger, disable_logger=True)
env = gym.wrappers.RecordVideo(env, video_folder=directory, disable_logger=True)

max_ep_len = 1000
device = "cpu"
scale = 1000.0  # normalization for rewards/returns
TARGET_RETURN = 3600.0 / scale  # evaluation is conditioned on a return of 3600, scaled accordingly

# mean and std computed from training dataset these are available in the model card for each model.
state_mean = np.array(
    [1.3490015,  -0.11208222, -0.5506444,  -0.13188992, -0.00378754,  2.6071432,
     0.02322114, -0.01626922, -0.06840388, -0.05183131,  0.04272673,]
)
state_std = np.array(
    [0.15980862, 0.0446214,  0.14307782, 0.17629202, 0.5912333,  0.5899924,
 1.5405099,  0.8152689,  2.0173461,  2.4107876,  5.8440027,]
)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Create the decision transformer model
model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert")
#model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
model = model.to(device)
print(list(model.encoder.wpe.parameters()))

state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)


# Interact with the environment and create a video
episode_return, episode_length = 0, 0
state, _ = env.reset()
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(0, device=device, dtype=torch.float32)

timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

for t in range(max_ep_len):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    actions[-1] = action
    action = action.detach().cpu().numpy()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1

    if done:
        break

env.close()

