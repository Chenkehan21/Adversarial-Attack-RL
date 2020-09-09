import gym
import time
import argparse
import numpy as np

import torch

from lib0 import wrappers
from lib0 import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25
# frame of second

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", required=True,
    #                     help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    # if args.record:
        # env = gym.wrappers.Monitor(env, args.record)
        # 第二个参数传入的是放到的文件的地址, 暂时先不用
    env = gym.wrappers.Monitor(env, "recording")

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n)
    state = torch.load(".\PongNoFrameskip-v4-best_19.dat", map_location=lambda stg, _: stg)
    # by default, torch save the cpu, gpu version of the tensor if you use gpu on training
    # map_location is needed to map the loaded tensor location from gpu to cpu
    # if you train on a cpu map_location is not needed

    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    # counter 集合中的元素 并以字典的形式返回

    while True:
        start_ts = time.time()
        #if args.vis:
        #   env.render()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        # 注意这里要使用[state] dim 会有问题
        q_vals = net(state_v).data.numpy()[0]
        # [[v1, v2, v3, v4, v5]]
        action = np.argmax(q_vals)
        # argmax 返回 index
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        #if args.vis:
        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    #if args.record:
    env.env.close()
