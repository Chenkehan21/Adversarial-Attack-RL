import gym_compete.new_envs.multi_agent_env as MAE
import gym_compete.new_envs.kick_and_defend as KD
import os
import numpy as np

kwargs = {'agent_names': ['ant', 'ant', 'ant', 'ant'],
          'init_pos': [(-1, 0, 0.75), (1, 0, 0.75), (0, 1, 0.75), (0, -1, 0.75)]
          }
# instace = MAE.MultiAgentEnv(**kwargs)

print(os.path.join(
    os.path.dirname(__file__),
    "new_envs", "assets",
    "world_body.ant_body.ant_body.xml"
))

kwargs2 = {'agent_names': ['humanoid_kicker', 'humanoid_goalkeeper'],
           'scene_xml_path': os.path.join(
               os.path.dirname(os.path.abspath(__file__)), "gym_compete", "new_envs",
               "assets",
               "world_body_football.humanoid_body.humanoid_body.xml"
           ),
           'world_xml_path': os.path.join(
               os.path.dirname(os.path.abspath(__file__)), "gym_compete", "new_envs",
               "assets", 'world_body_football.xml'
           ),
           'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
           'max_episode_steps': 500,
           }


def _random_actions(obs, agents):
    actions = []
    for i, agent in agents.items:
        actions.append(agent.action_space.sample())
    return actions


def simulation(env: KD.KickAndDefend):
    env.reset()
    while True:
        actions = _random_actions(None, env.agents)
        obs, reward, done, _ = env.step(actions)
        env.render()
        if done:
            break


def get_info(env: KD.KickAndDefend):
    n_agents = env.num_agents
    print("agent_nums = {}").format(n_agents)
    obs = env.reset
    print("obs dtype = {}").format(type(obs))


def main():
    instace = KD.KickAndDefend(**kwargs2)
    simulation(instace)


if __name__ == "__main__":
    main()


class a(KD.KickAndDefend):
    def __init__(self):
        super(a, self).__init__()

