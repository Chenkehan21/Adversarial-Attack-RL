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

    device = torch.device("cuda")

    writer = SummaryWriter()

    env = wrappers.make_env(args.env)
    # if args.record:
    # env = gym.wrappers.Monitor(env, args.record)
    # 第二个参数传入的是放到的文件的地址, 暂时先不用
    # env = gym.wrappers.Monitor(env, "recording")

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    state = torch.load(".\PongNoFrameskip-v4-best_14.dat", map_location=lambda stg, _: stg)
    # net 是 gpu 参数是 CPU???
    # by default, torch save the cpu, gpu version of the tensor if you use gpu on training
    # map_location is needed to map the loaded tensor location from gpu to cpu
    # if you train on a cpu map_location is not needed

    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    # counter 集合中的元素 并以字典的形式返回

    ATTACK_FRAME_SKIP = 1
    # 每10个frame执行一次attack
    step_count = 0

    while True:
        step_count = step_count + 1
        start_ts = time.time()
        # if args.vis:
        #   env.render()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False)).to(device)
        # 注意这里要使用[state] dim 会有问题

        if step_count % ATTACK_FRAME_SKIP == 0:
            writer.add_image("img", vutils.make_grid(
                state_v.data.cpu()[0][0], normalize=True), step_count)
            state_v = FGSM_ATTACK(state=state_v, net=net, device=device)
            writer.add_image("attack", vutils.make_grid(
                state_v.data.cpu()[0][0], normalize=True), step_count)


        q_vals = net(state_v).cpu().data.numpy()[0]
        # [[v1, v2, v3, v4, v5]]
        action = np.argmax(q_vals)
        print(action)
        # argmax 返回 index
        c[action] += 1
        state, reward, done, _ = env.step(action)
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
