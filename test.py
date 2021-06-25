import argparse
from carracing import CarRacing
from utils import env_wrapper
from stable_baselines3.td3 import TD3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.zip` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-r', '--render', action='store_true', help='render the env or not')
    # parser.add_argument('-r', '--random', action='store_true', help='use random agent or not')
    args = parser.parse_args()

    wrapped_env = CarRacing(diff_road_col=True)
    env = env_wrapper(env=wrapped_env, frame_stack_num=3, skip_frame=2)

    model = TD3.load(args.model)

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