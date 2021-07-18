import argparse
import glob
from os import PathLike, path
import json
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from carracing import CarRacing
from utils import env_wrapper
from stable_baselines3.td3 import TD3
from stable_baselines3.ppo import PPO
from stable_baselines3.dqn import DQN

# find the config.json based on the zip file
def find_config_file(zip_path: PathLike):
    dirname = path.dirname(zip_path)
    pattern = path.join(dirname, "config.json")
    files = glob.glob(pattern)

    if len(files) > 0:
        return files[0]
    
    # search the upper folder then
    pattern = path.join(path.dirname(dirname), "config.json")

    return glob.glob(pattern)[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.zip` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-r', '--render', action='store_true', help='render the env or not')
    args = parser.parse_args()

    config_file_path = find_config_file(args.model)

    with open(config_file_path, "r") as f:
        config = json.load(f)
        env_kwargs, algo = config['env_kwargs'], config['algo']

    if 'action_space' in env_kwargs:
        env_kwargs['action_space'] = np.array(env_kwargs['action_space'])
    
    algo: BaseAlgorithm = eval(algo)

    wrapped_env = CarRacing(diff_road_col=True)
    # env = env_wrapper(env=wrapped_env,
    #                   frame_stack_num=4,
    #                   skip_frame=2,
    #                   use_discrete_action_space=True,
    #                   num_discrete_steering=10,
    #                   num_discrete_throttle=3)
    env = env_wrapper(env=wrapped_env, **env_kwargs, is_training=False)

    model = algo.load(args.model)

    env.set_render(True)

    reward_list = [0 for _ in range(args.episodes)]
    for i in range(args.episodes):
        obs = env.reset()
        
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            reward_list[i] += reward
    
    print(f"rewards: {[round(reward, 2) for reward in reward_list]}")
    print(f"mean reward: {sum(reward_list) / args.episodes}")