import random

import gym
import time
import argparse
import numpy as np

import torch

from lib0 import wrappers
from lib0 import dqn_model

import collections
import torch.nn as nn
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


# frame of second
# buffer 4 * 84 * 84
def FGSM_ATTACK(state, net, epislon=0.01, device=torch.device("cuda")):
    state = state.to(device)
    state.requires_grad = True
    output = net(state)
    # output = nn.Softmax(dim=1)(output)
    target = torch.tensor([[0, 0, 0, 0, 2, 0]], dtype=torch.float).to(device)
    Iter_no = 0
    Iter_max_no = 5
    # while np.argmax(output.cpu().data.numpy()[0]) != 5 or Iter_no < Iter_max_no:
    while Iter_no < Iter_max_no:
        Iter_no = Iter_no + 1
        # print(torch.max(output))
        loss = nn.MSELoss()(output, target)
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            state_grad = state.grad.data
            sign_data_grad = state_grad.sign()
            Noise = epislon * sign_data_grad
            state = state + Noise[0][0]
            # print(state.is_leaf)
        # 注意这里要用with no grad 来避免state反复加入计算图中
        state.requires_grad = True
        output = net(state)
        # output = nn.Softmax(dim=1)(output)
    return state


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
    writer = SummaryWriter()

    from lib0.adversarial_env import adversarial_env

    env = wrappers.make_env(args.env)
    victim_net = dqn_model.DQN(env.observation_space.shape,
                               env.action_space.n).to(device)
    state = torch.load(".\PongNoFrameskip-v4-best_0.dat", map_location=lambda stg, _: stg)
    victim_net.load_state_dict(state)
    env = adversarial_env(env, victim_net=victim_net)
    env.seed(SEED)

    ad_net = dqn_model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device)
    state = torch.load(".\PongNoFrameskip-v4-ad-best_-201.dat", map_location=lambda stg, _: stg)
    ad_net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    step_count = 0

    while True:
        step_count = step_count + 1
        start_ts = time.time()
        env.render()

        state_v = torch.tensor(np.array([state], copy=False)).to(device)

        q_vals = ad_net(state_v).cpu().data.numpy()[0]
        action = int(np.argmax(q_vals))
        print(action)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        if action > 3:
            writer.add_scalar("Activate", 1, step_count)
            writer.add_image("activate_img", vutils.make_grid(
                state_v.data.cpu()[0][0], normalize=True), step_count)
            writer.add_image("ad_activate_img", vutils.make_grid(
                env.ad_state.data.cpu()[0][0], normalize=True), step_count)
        else:
            writer.add_scalar("Activate", 0, step_count)
        c[action] += 1
        # state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        # if args.vis:
        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    # if args.record:
    env.env.close()
