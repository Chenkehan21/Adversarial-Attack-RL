import gym
import numpy as np
import torch
import torch.nn as nn
from gym import spaces



class adversarial_env(gym.Env):
    metadata = {'render.model': ['human']}
    ACTION_PENALTY = -0.5

    def __init__(self, env, victim_net):
        super(adversarial_env, self).__init__()

        self.victim_net = victim_net
        self.env = env
        self.state = None
        self.ad_state = None

        self.action_space = spaces.Discrete(2*3)
        # 0: no attack + up     2
        # 1: no attack + down   5
        # 2: no attack + nope   0
        # 3: attack + up
        # 4: attack + down
        # 5: attack + nope

        self.observation_space = self.env.observation_space

        # self.Threshold = 1

    def reset(self):
        obs = self.env.reset()
        self.state = obs
        return obs

    def step(self, action: int, device=torch.device("cuda")):
        target_action = None
        if action < 3:
            state = torch.tensor([self.state], dtype=torch.float).to(device)
            with torch.no_grad():
                q_vals = self.victim_net(state).cpu().data.numpy()[0]
            action = np.argmax(q_vals)
            obs, r, done, info = self.env.step(action)
            self.state = obs
            return obs, -r, done, info
        else:
            if action == 3:
                target_action = [2]
            elif action == 4:
                target_action = [5]
            elif action == 5:
                target_action = [0]
            noise_state = self._FGSM_attack(state=self.state, net=self.victim_net, target_action=target_action)
            # print(noise_state)
            self.ad_state = noise_state
            q_vals = self.victim_net(noise_state).cpu().data.numpy()[0]
            action = np.argmax(q_vals)
            obs, r, done, info = self.env.step(action)
            r = -r
            r += self.ACTION_PENALTY
            self.state = obs
            return obs, r, done, info

    # def _action_penalty(self, action):
    #     return -1

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

    def _FGSM_attack(self, state, net, target_action, Iter_max_nu=4, epislon=0.001, device=torch.device("cuda")):
        if target_action is None:
            print("target_action = None")
            return
        state = torch.tensor([state], dtype=torch.float).to(device)
        state = state.to(device)
        state.requires_grad = True
        output = net(state)
        output = nn.Softmax(dim=1)(output)
        target_action = torch.tensor(target_action, dtype=torch.long).to(device)
        Iter_nu = 0
        Iter_max_nu = Iter_max_nu
        # while np.argmax(output.cpu().data.numpy()[0]) != 5 or Iter_nu < Iter_max_nu:
        while Iter_nu < Iter_max_nu:
            Iter_nu = Iter_nu + 1
            # print(torch.max(output))
            loss = -nn.CrossEntropyLoss()(output, target_action)
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
            output = nn.Softmax(dim=1)(output)
        return state

    def unwrapped(self):
        return self.env.unwrapped

    def seed(self, seed=None):
        self.env.seed(seed)
