import gym_compete.new_envs.multi_agent_env as MAE
import gym_compete.new_envs.kick_and_defend_v2 as KD
import os
import numpy as np

kwargs2 = {'agent_names': ['humanoid_kicker', 'humanoid_goalkeeper', 'humanoid_goalkeeper'],
           'scene_xml_path': os.path.join(
               os.path.dirname(os.path.abspath(__file__)), "gym_compete", "new_envs",
               "assets",
               "world_body_football.humanoid_body.humanoid_body.humanoid_body.xml"
           ),
           'world_xml_path': os.path.join(
               os.path.dirname(os.path.abspath(__file__)), "gym_compete", "new_envs",
               "assets", 'world_body_football.xml'
           ),
           'init_pos': [(-1, 0, 1.5), (1, -1, 1.5), (1, 1, 1.5)],
           'max_episode_steps': 500,
           }


def _random_actions(obs, agents):
    actions = []
    for i, agent in agents.items():
        actions.append(agent.action_space.sample())
    return actions


def simulation(env: KD.KickAndDefendV2):
    env.reset()
    while True:
        actions = _random_actions(None, env.agents)
        obs, reward, done, _ = env.step(actions)
        env.render()
        # if done:
        #     break


def get_info(env: KD.KickAndDefendV2):
    n_agents = env.num_agents
    print("agent_nums = {}").format(n_agents)
    obs = env.reset
    print("obs dtype = {}").format(type(obs))


def main():
    instace = KD.KickAndDefendV2(**kwargs2)
    simulation(instace)


if __name__ == "__main__":
    main()
