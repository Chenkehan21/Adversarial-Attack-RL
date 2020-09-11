from .multi_agent_env import MultiAgentEnv
import numpy as np
from .agents import Agent
import six
from gym import spaces

'''
kwargs={'agent_names': ['humanoid_kicker', 'humanoid_goalkeeper'], # ['humanoid_goalkeeper', 'humanoid_kicker']
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body_football.humanoid_body.humanoid_body.xml"
            ),
            'world_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets", 'world_body_football.xml'
            ),
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'max_episode_steps': 500,
            }
'''


class KickAndDefend(MultiAgentEnv):
    def __init__(self, max_episode_steps=500, randomize_ball=True, **kwargs):
        super(KickAndDefend, self).__init__(**kwargs)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        # steps state 使用
        self.GOAL_REWARD = 1000
        self.ball_jnt_id = self.env_scene.model.joint_names.index(six.b('ball'))
        self.jnt_nqpos = Agent.JNT_NPOS[int(self.env_scene.model.jnt_type[self.ball_jnt_id])]
        if self.agents[0].team == 'walker':
            self.walker_id = 0
            self.blocker_id = 1
        else:
            self.walker_id = 1
            self.blocker_id = 0
        # walker = kicker goalkeeper = blocker
        self.GOAL_X = self.agents[self.walker_id].TARGET
        # print("GOAL_X:", self.GOAL_X)
        self.GOAL_Y = 3
        self.randomize_ball = randomize_ball
        self.LIM_X = [(-3.5, -0.5), (1.6, 3.5)]
        # self.RANGE_X = self.LIM_X = [(-2, -2), (1.6, 3.5)]
        self.LIM_Y = [(-2, 2), (-2, 2)]
        self.RANGE_X = self.LIM_X.copy()
        self.RANGE_Y = self.LIM_Y.copy()
        self.BALL_LIM_X = (-2, 1)
        self.BALL_LIM_Y = (-4, 4)
        self.BALL_RANGE_X = self.BALL_LIM_X
        self.BALL_RANGE_Y = self.BALL_LIM_Y
        self.keeper_touched_ball = False

    def _past_limit(self):
        if self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def get_ball_qpos(self):
        start_idx = int(self.env_scene.model.jnt_qposadr[self.ball_jnt_id])
        return self.env_scene.model.data.qpos[start_idx:start_idx+self.jnt_nqpos]

    def get_ball_qvel(self):
        start_idx = int(self.env_scene.model.jnt_dofadr[self.ball_jnt_id])
        # ball has 6 components: 3d translation, 3d rotational
        return self.env_scene.model.data.qvel[start_idx:start_idx+6]

    def get_ball_contacts(self, agent_id):
        mjcontacts = self.env_scene.data._wrapped.contents.contact
        # env_scene.data = model.data
        ncon = self.env_scene.model.data.ncon
        # number of detected contacts
        contacts = []
        for i in range(ncon):
            ct = mjcontacts[i]
            g1, g2 = ct.geom1, ct.geom2
            g1 = self.env_scene.model.geom_names[g1]
            g2 = self.env_scene.model.geom_names[g2]
            if g1.find(six.b('ball')) >= 0:
                if g2.find(six.b('agent' + str(agent_id))) >= 0:
                    if ct.dist < 0:
                        contacts.append((g1, g2, ct.dist))
        return contacts
    # 逻辑：先list所有contacts 再从中检测和ball contact的属于agent id 的geoms
    # 不明白为什么dist是小于0判定？

    def _set_ball_xyz(self, xyz):
        start = int(self.env_scene.model.jnt_qposadr[self.ball_jnt_id])
        qpos = self.env_scene.model.data.qpos.flatten().copy()
        qpos[start:start+3] = xyz
        qvel = self.env_scene.model.data.qvel.flatten()
        self.env_scene.set_state(qpos, qvel)
    # 设置 ball xyz，joint位置可以唯一确定ball吗？三个自由度？

    def is_goal(self):
        ball_xyz = self.get_ball_qpos()[:3]
        if 0 < self.GOAL_X < ball_xyz[0] and abs(ball_xyz[1]) <= self.GOAL_Y:
            return True
        elif 0 > self.GOAL_X > ball_xyz[0] and abs(ball_xyz[1]) <= self.GOAL_Y:
            return True
        return False
    # 判断ball是否处于goal的位置

    def goal_rewards(self, infos=None, agent_dones=None):
        self._elapsed_steps += 1
        # print(self._elapsed_steps, self.keeper_touched_ball)
        goal_rews = [0. for _ in range(self.num_agents)]
        ball_xyz = self.get_ball_qpos()[:3]
        done = self._past_limit() or (0 < self.GOAL_X < ball_xyz[0]) or (0 > self.GOAL_X > ball_xyz[0])
        ball_vel = self.get_ball_qvel()[:3]
        if ball_vel[0] < 0 and np.linalg.norm(ball_vel) > 1:
            done = True
            # print("Keeper stopped ball, vel:", ball_vel)
        # agent_fallen = [self.agents[i].get_qpos()[2] < 0.5 for i in range(self.n_agents)]
        # import ipdb; ipdb.set_trace()
        ball_contacts = self.get_ball_contacts(self.blocker_id)
        if len(ball_contacts) > 0:
            # print("detected contacts for keeper:", ball_contacts)
            self.keeper_touched_ball = True
        if self.is_goal():
            for i in range(self.num_agents):
                if self.agents[i].team == 'walker':
                    goal_rews[i] += self.GOAL_REWARD
                    infos[i]['winner'] = True
                else:
                    goal_rews[i] -= self.GOAL_REWARD
            done = True
        elif done or all(agent_dones):
            for i in range(self.num_agents):
                if self.agents[i].team == 'walker':
                        goal_rews[i] -= self.GOAL_REWARD
                else:
                    goal_rews[i] += self.GOAL_REWARD
                    infos[i]['winner'] = True
                    if self.keeper_touched_ball:
                        # ball contact bonus
                        goal_rews[i] += 0.5 * self.GOAL_REWARD
                    if self.agents[i].get_qpos()[2] > 0.8:
                        # standing bonus
                        goal_rews[i] += 0.5 * self.GOAL_REWARD
        else:
            keeper_penalty = False
            for i in range(self.num_agents):
                if self.agents[i].team == 'blocker':
                    if np.abs(self.GOAL_X - self.agents[i].get_qpos()[0]) > 2.5:
                        keeper_penalty = True
                        break
            if keeper_penalty:
                done = True
                for i in range(self.num_agents):
                    if self.agents[i].team == 'blocker':
                        goal_rews[i] -= self.GOAL_REWARD
            else:
                for i in range(self.num_agents):
                    if self.agents[i].team == 'walker':
                        # goal_rews[i] -= np.abs(ball_xyz[0] - self.GOAL_X)
                        infos[i]['reward_move'] -= np.abs(ball_xyz[0] - self.GOAL_X).item()
                    else:
                        infos[i]['reward_move'] += np.abs(ball_xyz[0] - self.GOAL_X).item()
                        # if len(ball_contacts) > 0:
                        #     # ball contact bonus
                        #     print("detected contacts for keeper:", ball_contacts)
                        #     goal_rews[i] += 0.5 * self.GOAL_REWARD
        return goal_rews, done

    def _set_ball_vel(self, vel_xyz):
        start = int(self.env_scene.model.jnt_dofadr[self.ball_jnt_id])
        qvel = self.env_scene.model.data.qvel.flatten().copy()
        qvel[start:start+len(vel_xyz)] = vel_xyz
        qpos = self.env_scene.model.data.qpos.flatten()
        self.env_scene.set_state(qpos, qvel)

    def _set_random_ball_pos(self):
        x = self.env_scene.np_random.uniform(*self.BALL_RANGE_X)
        y = self.env_scene.np_random.uniform(*self.BALL_RANGE_Y)
        z = 0.35
        # print("setting ball to {}".format((x, y, z)))
        self._set_ball_xyz((x,y,z))
        if self.get_ball_qvel()[0] < 0:
            self._set_ball_vel((0.1, 0.1, 0.1))

    def _reset_range(self, version):
        decay_func = lambda x: 0.05 * np.exp(0.001 * x)
        v = decay_func(version)
        self.BALL_RANGE_X = (max(self.BALL_LIM_X[0], -v), min(self.BALL_LIM_X[1], v))
        self.BALL_RANGE_Y = (max(self.BALL_LIM_Y[0], -v), min(self.BALL_LIM_Y[1], v))
        self.RANGE_X[0] = (max(self.LIM_X[0][0], -2-v),  min(self.LIM_X[0][1], -2+v))
        self.RANGE_Y[0] = (max(self.LIM_Y[0][0], -v),  min(self.LIM_Y[0][1], v))
        self.RANGE_X[1] = (max(self.LIM_X[1][0], 2-v),  min(self.LIM_X[1][1], 2+v))
        self.RANGE_Y[1] = (max(self.LIM_Y[1][0], -v),  min(self.LIM_Y[1][1], v))
        # print(self.RANGE_X)
        # print(self.RANGE_Y)
        # print(self.BALL_RANGE_X)
        # print(self.BALL_RANGE_Y)

    def reset(self, margins=None, version=None):
        self._elapsed_steps = 0
        self.keeper_touched_ball = False
        _ = self.env_scene.reset()
        if version is not None:
            self._reset_range(version)
        for i in range(self.num_agents):
            x = self.env_scene.np_random.uniform(*self.RANGE_X[i])
            y = self.env_scene.np_random.uniform(*self.RANGE_Y[i])
            # print("setting agent {} to pos {}".format(i, (x,y)))
            self.agents[i].set_xyz((x, y, None))
            # 不知道这里为啥是None,会自动调整还是咋样
            self.agents[i].reset_agent()
        if self.randomize_ball:
            self._set_random_ball_pos()

        if self.agents[0].team == 'walker':
            self.walker_id = 0
            self.blocker_id = 1
        else:
            self.walker_id = 1
            self.blocker_id = 0
        self.GOAL_X = self.agents[self.walker_id].TARGET
        if margins is not None:
            for i in range(self.num_agents):
                self.agents[i].set_margin(margins[i])
        # print("GOAL_X:", self.GOAL_X)
        ob = self._get_obs()
        # tuple， 每个agent的obs
        return ob
