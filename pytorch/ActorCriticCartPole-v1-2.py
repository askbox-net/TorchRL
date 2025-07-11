"""
# 強化学習におけるActor-Criticの解説とPytorchによる実装

Actor-Critic法は方策ベース(Policy-based)と価値ベース(Value-based)の両方の利点を組み合わせた強化学習手法です。この記事では、Actor-Critic法の基本概念を解説し、PyTorchを使ってCartPole-v1環境での実装例を示します。

## 1. Actor-Criticの基本概念

Actor-Critic法は以下の2つの主要なコンポーネントから構成されています：

- **Actor**: 環境内での行動を決定する方策（policy）を学習
- **Critic**: 状態や状態-行動ペアの価値を評価し、Actorの学習を支援

### 利点

- 方策ベース手法の高い探索能力と、価値ベース手法の安定した学習を兼ね備える
- 方策勾配の分散を減少させる
- 連続的な行動空間にも対応可能

## 2. Actor-Criticのアルゴリズム

基本的なActor-Critic学習の流れ：

1. 現在の状態`s`を観測
2. Actorが方策`π`に基づいて行動`a`を選択
3. 行動`a`を環境に適用し、報酬`r`と次の状態`s'`を得る
4. Criticが状態の価値`V(s)`を推定
5. 時間差分（TD）誤差を計算: `δ = r + γV(s') - V(s)`
6. TD誤差を使ってActorとCriticを更新

## 3. PyTorchによるCartPole-v1の実装

それでは、PyTorchを使ってCartPole-v1環境でActor-Critic法を実装してみましょう。

## 4. コード解説

### ActorCriticネットワークの構造

- 単一の共有基盤層（fc1）を持ち、その上に2つのヘッドを持つ構造
  - **Actor**: 行動確率を出力する層（ソフトマックス関数適用）
  - **Critic**: 状態価値を推定する層

### 学習プロセス

1. 各エピソードで環境とのインタラクションを通じて経験を収集
2. 収集した報酬からリターン（割引報酬和）を計算
3. Criticが推定した状態価値とリターンの差からアドバンテージを計算
4. Actorの損失は「行動の対数確率 × アドバンテージ」の負値
5. Criticの損失は推定値と実際のリターンのMSE
6. 両方の損失を合わせて勾配計算と最適化

### その他の特徴

- 方策の確率分布にはCategorical分布を使用
- アドバンテージ関数を使って方策勾配の分散を減少
- 学習ループでは早期停止条件（直近10エピソードの平均報酬≧495）を設定

## 5. 発展的なトピック

実際のActor-Criticには様々な改良版があります：

- **A2C/A3C**: Advantage Actor-Critic、複数のエージェントによる並列学習
- **PPO**: Proximal Policy Optimization、方策更新を制約付きで行う
- **SAC**: Soft Actor-Critic、エントロピー正則化と複数のQ関数を使用

これらの手法は学習の安定性や効率をさらに向上させるものです。

以上が強化学習におけるActor-Critic法の基本概念とPyTorchによる実装です。
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

# 環境設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Criticネットワーク
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)

        # Actor（方策）ヘッド
        self.actor = nn.Linear(128, n_actions)

        # Critic（価値関数）ヘッド
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # 方策（行動確率）を出力
        action_probs = F.softmax(self.actor(x), dim=-1)

        # 状態価値を出力
        state_values = self.critic(x)

        return action_probs, state_values

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

# トレーニング関数
def train_actor_critic(env_name='CartPole-v1', gamma=0.99, lr=0.001, n_episodes=2000):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(input_dim, n_actions).to(device)

    if os.path.exists('actor_critic_cartpole.pth'):
        model.load_state_dict(torch.load('actor_critic_cartpole.pth', weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        state_values = []
        done = False
        episode_reward = 0

        while not done:
            # 行動の選択
            action, log_prob = model.act(state)

            # 環境を1ステップ進める
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 状態価値を取得
            state_tensor = torch.FloatTensor(state).to(device)
            _, value = model(state_tensor)

            # データの保存
            log_probs.append(log_prob.unsqueeze(0))
            rewards.append(reward)
            state_values.append(value)

            episode_reward += reward
            state = next_state
            done = terminated | truncated

        episode_rewards.append(episode_reward)

        # 損失計算と最適化
        optimizer.zero_grad()

        # リターン（割引報酬和）の計算
        R = 0
        returns = []

        for reward in rewards[::-1]:
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)

        #print(log_probs)
        #print(state_values)
        log_probs = torch.cat(log_probs)
        state_values = torch.cat(state_values)

        # アドバンテージの計算
        advantage = returns - state_values.detach()

        # Actorの損失（方策勾配）
        actor_loss = -(log_probs * advantage).mean()

        # Criticの損失（MSE）
        critic_loss = F.mse_loss(state_values, returns)

        # 全体の損失
        loss = actor_loss + critic_loss

        loss.backward()
        optimizer.step()

        # 学習進捗の表示
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")

        # 学習完了条件
        if np.mean(episode_rewards[-10:]) >= 1024:#495:
            print(f"環境は{episode}エピソードで解決されました!")
            break

    return model, episode_rewards

# トレーニングの実行
if __name__ == "__main__":
    model, rewards = train_actor_critic()

    # モデルの保存
    torch.save(model.state_dict(), "actor_critic_cartpole.pth")

    # 学習したモデルのテスト
    #env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action, _ = model.act(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"テスト報酬: {total_reward}")
    env.close()
