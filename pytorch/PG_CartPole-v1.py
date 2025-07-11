# https://zenn.dev/takesan150/articles/5e5e86638f4c3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

EPISODES = 500 # 総エピソード数
MAX_STEPS = 500 # 1エピソードでの最大ステップ数
LEARNING_RATE = 0.0002 # 学習率
DISCOUNT_RATE = 0.99 # 割引率
STATE_SIZE = 4 # 状態数
ACTION_SIZE = 2 # 行動数
LOG_EPISODES = 500 # ログ出力のステップ頻度

class Net(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x

from typing import Tuple, Union
import torch.optim as optim
from torch.distributions import Categorical

class Agent:
    def __init__(self, state_size: int, action_size: int):
        self.gamma = DISCOUNT_RATE
        self.lr = LEARNING_RATE

        self.state_size = state_size
        self.action_size = action_size

        self.memory = []
        self.net = Net(self.state_size, self.action_size)
        if os.path.exists('pg_net.pt'):
            self.net.load_state_dict(torch.load('model.pt'))
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state = torch.tensor(state[np.newaxis, :])
        probs = self.net(state)
        probs = probs[0]
        d = Categorical(probs)
        action = d.sample()
        log_prob = d.log_prob(action)

        return action.item(), log_prob

    def add_experience(self, reward: Union[int, float], log_prob: torch.Tensor) -> None:
        data = (reward, log_prob)
        self.memory.append(data)

    def update(self) -> None:
        gain, loss = 0, 0
        for reward, log_prob in reversed(self.memory):
            gain = reward + self.gamma * gain

        for reward, log_prob in self.memory:
            loss += -log_prob * gain

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
    
    def save_model(self) -> None:
        torch.save(self.net.state_dict(), 'model.pth')

import gymnasium as gym
from logging import getLogger, basicConfig, INFO

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

def main():
    # train
    env = gym.make('CartPole-v1')
    agent = Agent(STATE_SIZE, ACTION_SIZE)
    for e in range(EPISODES):
        done = False
        state, _ = env.reset()
        step = 0
        while step < MAX_STEPS:
            action, prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.add_experience(reward, prob)

            state = next_state
            step += 1

            if done:
                break
        if e % LOG_EPISODES == 0:
            logger.info("episodes: %d", e)
        agent.update()
    agent.save_model()
    env.close()
    # test
    env = gym.make('CartPole-v1', render_mode="human")
    done = False
    state, _  = env.reset()
    step = 0
    while step < MAX_STEPS:
        env.render()
        action, _ = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        if done:
            break
            
if __name__ == '__main__':
    main()      

