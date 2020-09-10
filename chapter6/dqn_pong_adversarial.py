import random

from lib0 import wrappers
from lib0 import dqn_model

import argparse
import time
import numpy as np
import collections
# collections 就是一系列的集合，改写 tuple 命名一些三元组之类的

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 5

GAMMA = 0.99  # degree of "death" or how eager your agent
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000  # k3 times the target net work update
REPLAY_START_SIZE = 10000
# AGENT_PROCESS = 100       # k1 times the agent collects data
# REGRESSION_EPOCH = 5      # K2 times epoch is a single pass through the full training set

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0  # epsilon decay, during the first 150,000 frames, epsilon
EPSILON_FINAL = 0.01  # is linearly decayed to 0.01

# core function 1 replay buffer
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 双端队列

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        # replace=False保证调出来的不一样
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        # * 解包 [(1),(2), (3)] => (1), (2), (3) 再将这些单独的“个体” 传入到zip(iter1, iter2, ...)参数中
        #  zip is an iterator of tuples where the first item in each passed iterator
        #  is paired together, and then the second item in each passed iterator are
        #  paired together etc.
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent(object):
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device: torch.device = torch.device("cuda")):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            # 注意 [self.state]
            state_v = torch.tensor(state_a).to(device)
            with torch.no_grad():
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
            # Returns a namedtuple (values, indices)
                action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device: torch.device = torch.device("cuda")):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    # 可能states 都太大所以 copy false 然后转成 torch？
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).long().to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    # unsqueeze(-1) -1 处再加一个维度
    # squeeze(): Returns a tensor with all the dimensions of input of size 1 removed.
    # index (LongTensor) 必须是和 input (Tensor) "上面" 同类的 tensor
    # 值储存的是对应的索引
    # [[1,2,3],[1,2,3],[1,2,3]]  [[1], [0], [2]] => [[2], [1], [3]]
    # squeeze(-1)   -1 squeeze最后一个维度(-1)  取消掉最后一个维度
    # the result of gather() applied to tensor is a differentiable operation

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        # tensor.max(dim = n) => (max_value, argmax)
        next_state_values[done_mask] = 0.0
        # without this training will not converge
        # 可能比如说最后一个state的值会摇摆, 由于generalize的原因
        next_state_values = next_state_values.detach()
        # no_grad 是暂时的detach()   .detach()是永久的detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)
    # mean squared error

SEED = 2

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda")

    from lib0.adversarial_env import adversarial_env

    env = wrappers.make_env(args.env)
    victim_net = dqn_model.DQN(env.observation_space.shape,
                               env.action_space.n).to(device)
    state = torch.load(".\PongNoFrameskip-v4-best_14.dat", map_location=lambda stg, _: stg)
    victim_net.load_state_dict(state)
    env = adversarial_env(env, victim_net=victim_net)
    env.seed(SEED)

    ad_net = dqn_model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device)
    ad_tgt_net = dqn_model.DQN(env.observation_space.shape,
                               env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(ad_net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(ad_net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0

    ts_frame = 0
    ts = time.time()
    # track our speed

    best_m_reward = None
    # best mean reward

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(ad_net, epsilon, device)
        # done_reward
        if reward is not None:
            # 结束了一个 eposide
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            # 不住100就按不足100的处理
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                      frame_idx, len(total_rewards), m_reward.item(), epsilon,
                      speed
                  ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(ad_net.state_dict(), args.env +
                           "-ad-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward.item(), m_reward.item()))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            ad_tgt_net.load_state_dict(ad_net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, ad_net, ad_tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    writer.close()
