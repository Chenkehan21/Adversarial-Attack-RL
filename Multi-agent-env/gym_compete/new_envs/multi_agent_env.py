import numpy as np
from gym import Env, spaces
from .multi_agent_scene import MultiAgentScene
from .agents import *
from .utils import create_multiagent_xml
import os
import six

class MultiAgentEnv(Env):
    '''
    A multi-agent environment consists of some number of Agent and
    a MultiAgentScene
    The supported agents and their classes are defined in
    AGENT_MAP, a dictionary mapping {agent_name: (xml_path, class)}
    Agents with initial x coordinate < 0 have goal on the right and
    vice versa
    '''
    AGENT_MAP = {
        'ant': (
            os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml"),
            Ant
        ),
        'humanoid': (
            os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml"),
            Humanoid
        ),
        'humanoid_blocker': (
            os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml"),
            HumanoidBlocker
        ),
        'humanoid_fighter': (
            os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml"),
            HumanoidFighter
        ),
        'ant_fighter': (
            os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml"),
            AntFighter
        ),
        'humanoid_kicker': (
            os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml"),
            HumanoidKicker
        ),
        'humanoid_goalkeeper': (
            os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml"),
            HumanoidGoalKeeper
        ),
    }
    WORLD_XML = os.path.join(os.path.dirname(__file__), "assets", "world_body.xml")
    GOAL_REWARD = 1000

    def __init__(
        self, agent_names,
        world_xml_path=WORLD_XML, agent_map=AGENT_MAP,
        scene_xml_path=None, move_reward_weight=1.0,
        init_pos=None, rgb=None, agent_args=None
    ):
        '''
            agent_args is a list of kwargs for each agent
        '''
        self.num_agents = len(agent_names)
        self.agents = {}
        all_agent_xml_paths = []
        if not agent_args:
            agent_args = [{} for _ in range(self.num_agents)]
        assert len(agent_args) == self.num_agents, "Incorrect length of agent_args"
        for i, name in enumerate(agent_names):
            print("Creating agent", name)
            agent_xml_path, agent_class = agent_map[name]
            self.agents[i] = agent_class(i, agent_xml_path, **agent_args[i])
            all_agent_xml_paths.append(agent_xml_path)
        agent_scopes = ['agent' + str(i) for i in range(self.num_agents)]
        # 设置了每个agent
        # print(scene_xml_path)
        if scene_xml_path is not None and os.path.exists(scene_xml_path):
            self._env_xml_path = scene_xml_path
        else:
            print("Creating Scene XML")
            print(init_pos)
            _, self._env_xml_path = create_multiagent_xml(
                world_xml_path, all_agent_xml_paths, agent_scopes,
                outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
                outpath=scene_xml_path,
                ini_pos=init_pos, rgb=rgb
            )
        # 创建multiagent xml
        print("Scene XML path:", self._env_xml_path)
        self.env_scene = MultiAgentScene(self._env_xml_path, self.num_agents)
        print("Created Scene with agents")
        for i, agent in self.agents.items():
            agent.set_env(self.env_scene)
        self._set_observation_space()
        self._set_action_space()
        self.metadata = self.env_scene.metadata
        self.move_reward_weight = move_reward_weight
        gid = self.env_scene.model.geom_names.index(six.b('rightgoal'))
        # 有的这个不可见 左右红线x位置
        self.RIGHT_GOAL = self.env_scene.model.geom_pos[gid][0]
        gid = self.env_scene.model.geom_names.index(six.b('leftgoal'))
        self.LEFT_GOAL = self.env_scene.model.geom_pos[gid][0]
        for i in range(self.num_agents):
            if self.agents[i].get_qpos()[0] > 0:
                self.agents[i].set_goal(self.LEFT_GOAL)
            else:
                self.agents[i].set_goal(self.RIGHT_GOAL)
        # set goal ?

    def _set_observation_space(self):
        self.observation_space = spaces.Tuple(
            [self.agents[i].observation_space for i in range(self.num_agents)]
        )

    def _set_action_space(self):
        self.action_space = spaces.Tuple(
            [self.agents[i].action_space for i in range(self.num_agents)]
        )

    def goal_rewards(self, infos=None, agent_dones=None):
        touchdowns = [self.agents[i].reached_goal()
                      for i in range(self.num_agents)]
        # 为什么会有return goal的判定？
        num_reached_goal = sum(touchdowns)
        goal_rews = [0. for _ in range(self.num_agents)]
        if num_reached_goal != 1:
            return goal_rews, num_reached_goal > 0
        #
        for i in range(self.num_agents):
            if touchdowns[i]:
                goal_rews[i] = self.GOAL_REWARD
                if infos:
                    infos[i]['winner'] = True
            else:
                goal_rews[i] = - self.GOAL_REWARD
        return goal_rews, True

    def _get_done(self, dones, game_done):
        done = np.all(dones)
        done = game_done or not np.isfinite(self.state_vector()).all() or done
        return bool(done)

    def step(self, actions):
        for i in range(self.num_agents):
            self.agents[i].before_step()
        self.env_scene.simulate(actions)
        move_rews = []
        infos = []
        dones = []
        for i in range(self.num_agents):
            move_r, agent_done, rinfo = self.agents[i].after_step(actions[i])
            move_rews.append(move_r)
            dones.append(agent_done)
            rinfo['agent_done'] = agent_done
            infos.append(rinfo)
        goal_rews, game_done = self.goal_rewards(infos=infos, agent_dones=dones)
        rews = []
        for i, info in enumerate(infos):
            info['reward_remaining'] = float(goal_rews[i])
            rews.append(float(goal_rews[i] + self.move_reward_weight * move_rews[i]))
        rews = tuple(rews)
        done = self._get_done(dones, game_done)
        infos = {i: info for i, info in enumerate(infos)}
        # 1： info   2: info    3: info
        obses = self._get_obs()
        return obses, rews, done, infos

    def _get_obs(self):
        return tuple([self.agents[i]._get_obs() for i in range(self.num_agents)])

    '''
    Following remaps all mujoco-env calls to the scene
    '''
    def seed(self, seed=None):
        return self.env_scene.seed(seed)

    def reset(self):
        # _ = self.env_scene.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        self.env_scene.set_state(qpos, qvel)

    @property
    def dt(self):
        return self.env_scene.dt

    def render(self, mode='human', close=False):
        return self.env_scene.render(mode, close)

    def state_vector(self):
        return self.env_scene.state_vector()

    def reset_model(self):
        # self.env_scene.reset_model()
        _ = self.env_scene.reset()
        for i in range(self.num_agents):
            self.agents[i].reset_agent()
            #更改agent的goal
        return self._get_obs()

    def viewer_setup(self):
        self.env_scene.viewer_setup()
