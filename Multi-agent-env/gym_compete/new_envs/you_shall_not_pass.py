from .multi_agent_env import MultiAgentEnv
import numpy as np

'''
register(
    id='multicomp/YouShallNotPassHumans-v0',
    entry_point='gym_compete.new_envs:HumansBlockingEnv',
    kwargs={'agent_names': ['humanoid_blocker', 'humanoid'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body.humanoid_body.humanoid_body.xml"
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)],
            'max_episode_steps': 500,
            },
)
'''


class HumansBlockingEnv(MultiAgentEnv):
    '''
        Two teams: walker and blocker
        Walker needs to reach the other end and bloker need to block them
        Rewards:
            Some Walker reaches end:
                walker which did touchdown: +1000
                all blockers: -1000
            No Walker reaches end:
                all walkers: -1000
                if blocker is standing:
                    blocker gets +1000
                else:
                    blocker gets 0
        NOTE: walker is fallen if z < 0.3
    '''

    def __init__(self, max_episode_steps=500, **kwargs):
        super(HumansBlockingEnv, self).__init__(**kwargs)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.GOAL_REWARD = 1000

    def _past_limit(self):
        if self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def _is_standing(self, agent_id, limit=0.3):
        return bool(self.agents[agent_id].get_qpos()[2] > limit)

    def _get_done(self, dones, game_done):
        return bool(game_done)

    def goal_rewards(self, infos=None, agent_dones=None):
        self._elapsed_steps += 1
        goal_rews = [0. for _ in range(self.num_agents)]
        touchdowns = [self.agents[i].reached_goal()
                      for i in range(self.num_agents)]
        # blockers 会直接是none的结果
        walkers_fallen = [not self._is_standing(i)
                          for i in range(self.num_agents)
                          if self.agents[i].team == 'walker']

        # print(self._elapsed_steps, touchdowns, walkers_fallen)

        done = self._past_limit() or all(walkers_fallen)
        # print(self._elapsed_steps,touchdowns, walkers_fallen)
        if not any(touchdowns):
            all_walkers_fallen = all(walkers_fallen)
            # The all() function returns True if all items in an iterable are true,
            # otherwise it returns False.
            # game_over = all_walkers_fallen
            for j in range(self.num_agents):
                if self.agents[j].team == 'blocker':
                    # goal_rews[j] += -infos[1-j]['reward_goal_dist']
                    infos[j]['reward_move'] += -infos[1-j]['reward_goal_dist']
                    # 只有两个才用的是 1
                if all_walkers_fallen and self.agents[j].team == 'blocker':
                    if self._is_standing(j):
                        goal_rews[j] += self.GOAL_REWARD
                    infos[j]['winner'] = True
                elif done:
                    if self.agents[j].team == 'walker':
                        goal_rews[j] -= self.GOAL_REWARD
                    else:
                        infos[j]['winner'] = True
        else:
            # some walker touched-down
            done = True
            for i in range(self.num_agents):
                if self.agents[i].team == 'walker':
                    if touchdowns[i]:
                        goal_rews[i] += self.GOAL_REWARD
                        infos[i]['winner'] = True
                else:
                    goal_rews[i] -= self.GOAL_REWARD

        # print(done, self._elapsed_steps, self._past_limit())
        return goal_rews, done

    def reset(self, margins=None):
        self._elapsed_steps = 0
        ob = super(HumansBlockingEnv, self).reset()
        if margins:
            for i in range(self.num_agents):
                self.agents[i].set_margin(margins[i])
        return ob
