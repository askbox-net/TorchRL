"""
以下に PyTorch を使用した Actor-Critic アルゴリズムによる CartPole-v1 環境の実装を示します。

## コードの解説:

1. **ActorCritic クラス**:
   - 状態を入力として受け取り、Actor（行動確率分布）と Critic（状態価値）を出力する共有ネットワーク
   - Actor は行動の確率分布を出力し、Critic は状態の価値を予測

2. **ActorCriticAgent クラス**:
   - ネットワークとオプティマイザーの管理
   - エピソードの経験を蓄積し、パラメータ更新を行う
   - Advantage を計算して Actor と Critic の損失を算出

3. **学習ループ**:
   - 各エピソードで行動を選択し、環境とのインタラクションを行う
   - エピソード終了後にネットワークパラメータを更新
   - 50エピソードの平均スコアが475を超えたら学習成功とみなす

このコードは、Actor-Critic アルゴリズムの基本的な実装です。Actor は行動選択のための確率分布を学習し、Critic はその行動の価値を評価します。両者が協力することで効率的な学習が可能になります。

CartPole-v1 環境は比較的簡単な問題なので、数百エピソード程度で解決できるでしょう。環境によって学習のハイパーパラメータを調整する必要があります。
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 再現性のために乱数シードを設定
torch.manual_seed(0)
np.random.seed(0)

# Actor-Critic ネットワークの定義
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 共有層
        self.fc1 = nn.Linear(state_dim, 128)

        # Actor (方策)
        self.actor = nn.Linear(128, action_dim)

        # Critic (価値関数)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # Actor: 行動の確率分布を出力
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic: 状態価値を出力
        state_value = self.critic(x)

        return action_probs, state_value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

# エージェントクラス
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def select_action(self, state):
        action, log_prob = self.model.act(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        _, value = self.model(state)

        #self.log_probs.append(log_prob)
        self.log_probs.append(log_prob.unsqueeze(0))
        #self.values.append(value.squeeze())
        self.values.append(value)

        return action

    def update(self):
        returns = []
        R = 0

        # リターンの計算
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = reward + self.gamma * R * mask
            returns.insert(0, R)

        returns = torch.tensor(returns).unsqueeze(1)

        # 損失の計算
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)

        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + critic_loss

        # パラメータの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # メモリをクリア
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

        return loss.item()

# 学習
def train(env_name="CartPole-v1", num_episodes=1000):
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim)

    max_steps = 500

    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated | truncated
            agent.rewards.append(reward)
            agent.masks.append(1 - done)

            state = next_state
            score += reward

            if done:
                break

        loss = agent.update()
        scores.append(score)

        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Loss: {loss:.4f}')

        # 終了条件（平均スコアが475以上）
        if len(scores) >= 50 and np.mean(scores[-50:]) >= 475:
            print(f'Environment solved in {episode} episodes!')
            break

    env.close()
    return scores

if __name__ == "__main__":
    scores = train()

    # 最後の結果を表示
    print(f'Final 50 episodes average score: {np.mean(scores[-50:]):.2f}')

