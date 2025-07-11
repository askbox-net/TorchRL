import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 再現性のためにシード値を設定
torch.manual_seed(42)
np.random.seed(42)

# 方策ネットワークの定義
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

# 方策勾配法のエージェント
class PolicyGradientAgent:
    def __init__(self, input_dim, output_dim):
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        action, log_prob = self.policy.select_action(state)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        # リターンを計算
        R = 0
        returns = []
        # 時間的な逆順で報酬を処理
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # 0.99は割引率γ
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # リターンの正規化
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 方策勾配の計算
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)  # 勾配上昇法のため、マイナスをつける

        policy_loss = torch.cat(policy_loss).sum()

        # パラメータ更新
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # メモリをクリア
        self.log_probs = []
        self.rewards = []

# トレーニング関数
def train(env, agent, n_episodes=1000):
    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            score += reward

        agent.update()
        scores.append(score)

        # 100エピソードごとに平均スコアを表示
        if episode % 100 == 0:
            print(f'エピソード {episode}, 平均スコア: {np.mean(scores[-100:]):.2f}')

        # 環境が解決されたと判断する条件
        if np.mean(scores[-100:]) >= 475.0:
            print(f'環境が{episode}エピソードで解決されました！')
            break

    return scores

# メイン関数
def main():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PolicyGradientAgent(input_dim, output_dim)

    scores = train(env, agent)

    # 学習済みモデルのテスト
    print("\n学習済みモデルのテスト")
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward

    print(f'テストスコア: {score}')
    env.close()

if __name__ == "__main__":
    main()
