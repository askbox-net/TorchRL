
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque

# --- ハイパーパラメータ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 1e-4         # Actorの学習率
LR_CRITIC = 1e-3        # Criticの学習率
GAMMA = 0.99            # 割引率
TAU = 1e-3              # ターゲットネットワークのソフトアップデート率
BUFFER_SIZE = int(1e6)  # リプレイバッファのサイズ
BATCH_SIZE = 128        # バッチサイズ
MAX_EPISODES = 1000     # 最大エピソード数
MAX_STEPS = 500         # 1エピソードの最大ステップ数
NOISE_STDDEV = 0.2      # OUノイズの標準偏差

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.FloatTensor(np.array(actions)).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# --- OUノイズ ---
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# --- Actorネットワーク ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.tanh(self.layer_3(x)) * self.max_action
        return x

# --- Criticネットワーク ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# --- DDPGエージェント ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.noise = OUNoise(action_dim, sigma=NOISE_STDDEV)
        self.max_action = max_action

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if add_noise:
            action += self.noise.sample()
        return action.clip(-self.max_action, self.max_action)

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # --- Criticの更新 ---
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (GAMMA * target_Q * (1 - dones)).detach()

        current_Q = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actorの更新 ---
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- ターゲットネットワークのソフトアップデート ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))


# --- 学習ループ ---
def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    total_steps = 0
    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            agent.update()

            if done:
                break
        
        print(f"Episode: {i_episode}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

    env.close()
    agent.save("DDPG_Pendulum-v1")

if __name__ == '__main__':
    main()
