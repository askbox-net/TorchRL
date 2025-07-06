# https://www.kaggle.com/code/maulberto3/cartpole-v1-pytorch-rl-actor-critic
import sys
import time
from collections import namedtuple
from pprint import pprint

import gym
import numpy as np
import torch as pt
import torch.distributions as dist
import torch.nn as nn
import os

# import pandas as pd

GAMMA = 0.9999 # 0.99
LR = 0.0025 # 0.01
EPISODES_TO_TRAIN = 15 # 10
HIDDEN_SIZE = 256 # 128
BELLMAN_STEPS = 4
BETA_ACTOR, BETA_CRITIC, BETA_ENTROPY = 1, 0.001, 1, 

EpisodeStep = namedtuple('EpisodeStep', field_names=[
                            'state',
                            'action',
                            'value',
                            'log_prob',
                            'entropy',
                            'reward',
                            'next_state',
                            'next_value',
                            'is_done'
                            ])


class ActorCritic(nn.Module):
    def __init__(self, inputs_features, n_actions, hidden_size):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(inputs_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(inputs_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        distr  = dist.Categorical(probs)
        return distr, value


class Agent():
    def __init__(self, env, disc_factor, net, optimizer, scheduler):
        self.env = env
        self.first_state = env.reset
        self.disc_factor = disc_factor
        self.net = net
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.distr = ''

    def explore(self):
        episodes = []
        state = self.first_state()
        state = pt.from_numpy(state).float()
        while True:
            # SA...
            distr, value = self.net.forward(state) # value with grads
            action = distr.sample()
            log_prob = distr.log_prob(action) # with grads
            entropy = distr.entropy().mean() # with grads
            
            # RS'...
            next_state, reward, is_done, _ = self.env.step(action.item())
            # V
            next_state = pt.from_numpy(next_state).float()
            _, next_value = self.net.forward(next_state) # next_value with grads
            
            # store results
            episodes.append(EpisodeStep(state=state,
                                        action=action,
                                        value=value,
                                        log_prob=log_prob,
                                        entropy=entropy,
                                        reward=reward,
                                        next_state=next_state,
                                        next_value=next_value,
                                        is_done=is_done))
            if is_done:
                yield episodes
                self.distr = distr
                episodes = []
                state = self.first_state()
                state = pt.from_numpy(state).float()
            state = next_state

    def calc_qvals(self, rewards):
        w = []
        sum_r = 0.0
        for reward in reversed(rewards):
            sum_r *= self.disc_factor
            sum_r += reward
            w.append(sum_r)
        w = list(reversed(w))
        w = pt.tensor(w)
        # normalize
        w = w - w.quantile(0.5)
        # w = (w - w.mean()) / w.std()
        # w = w.tolist()
        return w
    
    def calc_qvals_plus_v2(self, rewards, next_values, bellman_steps):
        result = []
        steps = [bellman_steps] * len(rewards)
        for i in range(len(rewards)):
            # ...4 4 4 4 4 3 2 1
            if i + steps[i] > len(rewards):
                take_step = i + steps[i] - len(rewards)
                steps[i] = steps[i] - take_step
        for step, rew in zip(steps, range(len(rewards))):
            # Q = sum_i_n(gamma^n*r_i) + gamma^n*V using just belllman steps
            if step >= bellman_steps:
                q = self.calc_qvals(rewards[rew: rew + step])
                q = q[0]
                q_v = q + GAMMA ** bellman_steps * next_values[rew + step - 1]
                result.append(q_v)
            else:
                q = self.calc_qvals(rewards[rew: rew + step])[0]
                result.append(q)
        return result

    def learn(self):
        global EPISODES_TO_TRAIN
        states, actions, rewards, values, next_states, next_values = [], [], [], [], [], []
        entropys, log_probs, qvals, accum_rewards, q_plus_vss, advs = [], [], [], [], [], []
        sum_rewards, base_qvals, scales = 0.0, 0.0, 0.0
        episodes, loops, best_episodes = 0, 0, 0
        
        for episode, explored in enumerate(self.explore()):
            # EXPLORE
            states.extend([x.state for x in explored])
            actions.extend([x.action for x in explored])
            log_probs.extend([x.log_prob for x in explored])
            values.extend([x.value for x in explored])
            values_copy = values.copy()
            entropys.extend([x.entropy for x in explored])
            rewards.extend([x.reward for x in explored])
            next_states.extend([x.next_state for x in explored])
            next_values.extend([x.next_value for x in explored])
            next_values_copy = next_values.copy()
            # steps.extend([[obs.step for obs in explored][-1]]) # just # steps at the last episode

            qvals.extend(self.calc_qvals(rewards))
            # advs.extend(adv)
            
            sum_rewards = sum(rewards)
            rewards.clear()
            next_values.clear()
            values.clear()
            # q_plus_vs, adv = 0, 0
            episodes += 1

            # CHECK Solution
            accum_rewards.append(sum_rewards)
            mean_100th_rew = np.mean(accum_rewards[-100:])
            if loops <= 100 and mean_100th_rew > 110:
                print(f"Solved in {episode} episodes, {loops} loops!")
                break
            if episodes < EPISODES_TO_TRAIN:
                continue  # continue yielding episodes and...
            # ...train net for better action prediction

            # Critic Loss
            # train_state_state = pt.stack(states)
            # distr, value = self.net.forward(train_state_state) # .to('cuda', non_blocking=True)
            critic_loss = 0.5 * pt.mean((pt.stack(qvals) - pt.stack(values_copy))**2) #

            # Actor Loss
            actor_loss = pt.stack(log_probs) * pt.stack(qvals) # .to('cuda', non_blocking=True)
            actor_loss = pt.mean(actor_loss)

            # Entropy loss
            entropy_loss = pt.mean(self.distr.entropy())

            # Full loss
            loss = BETA_ACTOR * -actor_loss + BETA_CRITIC * critic_loss + BETA_ENTROPY * -entropy_loss
            loss_detail = (BETA_ACTOR * -actor_loss.detach().item(), BETA_CRITIC * critic_loss.detach().item(), BETA_ENTROPY * -entropy_loss.detach().item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), 5) # security grad explosion
            self.optimizer.step()
            self.scheduler.step() # reduce lr by x amount

            # Batch info (mean): s, a, r, s', q,
            print(f'Loop: {loops} | Episode: {episode} | Last 100 Rew: {mean_100th_rew:,.2f} | Loss: {loss:+,.3f} ( {loss_detail[0]:,.2f} {loss_detail[1]:,.2f} {loss_detail[2]:,.2f} ) ')

            # Saving model
            # crc_tz = timezone("America/Costa_Rica")
            # now = datetime.now(tz=crc_tz)
            # now = datetime.strftime(now, "%b_%d_%Y")
            # if episodes < best_episodes:
            #     pt.save(self.net.state_dict(), f'cartpole_rl_agent_{now}_episodes_{episodes}.pt')
            #     best_episodes = episodes

            episodes = 0
            EPISODES_TO_TRAIN += 1
            states, actions, log_probs, values, entropys = [], [], [], [], []
            states, actions, q_plus_vss, advs, next_values, qvals = [], [], [], [], [], []
            loops += 1

if __name__ == "__main__":

    ####################################
    # Setups
    print('*'*50)

    ####################################
    # Environment
    env = gym.make("CartPole-v1")

    ####################################
    # Neural Networks
    actor_critic = ActorCritic(inputs_features=env.observation_space.shape[0],
                                n_actions=env.action_space.n,
                                hidden_size=HIDDEN_SIZE,
                                    )
    model_name = ''
    if os.path.exists(model_name):
        actor_critic.load_state_dict(pt.load(model_name))
    actor_critic = actor_critic.float()
    ac_optimizer = pt.optim.Adam(actor_critic.parameters(), lr=LR) # , lr=LR Adam AdamW Adamax
    lmbda = lambda epoch: 0.99
    ac_scheduler = pt.optim.lr_scheduler.MultiplicativeLR(ac_optimizer, lr_lambda=lmbda)
    
    ####################################
    # Agent
    agent = Agent(env, GAMMA, actor_critic, ac_optimizer, ac_scheduler)
    agent.learn()
    
    pt.save(actor_critic.state_dict(), model_name)

