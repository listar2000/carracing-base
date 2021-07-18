from stable_baselines3.ppo import PPO
from stable_baselines3.dqn import DQN
# from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.dqn.policies import CnnPolicy
# from stable_baselines3.td3 import TD3
# from stable_baselines3.td3.policies import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, EveryNTimesteps, CallbackList
from utils import LoggingCallback, env_wrapper, AC_SPACE_V2, discretize_action_space
from model import CustomMaxPoolCNN
from pathlib import Path
from carracing import CarRacing
import torch as th
import os
from datetime import datetime
import json

if __name__ == '__main__':
    wrapped_env = CarRacing(diff_road_col=True)

    custom_action_space = discretize_action_space([-1, -0.5, 0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1])

    if th.cuda.is_available():
        print("using cuda:0 right now...")
        device = th.device('cuda:0')
    else:
        print("no cuda available, cpu used for training...")
        device = th.device('cpu')

    env_kwargs = dict(
        frame_stack_num=3,
        skip_frame=2,
        action_space_type="discrete",
        action_space=custom_action_space.tolist(),
    )

    env = env_wrapper(env=wrapped_env, **env_kwargs)
    env.set_render(every=10)

    policy_kwargs = dict(
        features_extractor_class=CustomMaxPoolCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )

    training_kwargs = dict(
        learning_rate=0.001,
        buffer_size=5000,
        batch_size=64,
        learning_starts=10000,
        gamma=0.95,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        seed=1
    )

    # specify the algorithm to use here...
    rl_algo = DQN

    # model = PPO(CnnPolicy, env,
    #             device=device,
    #             learning_rate=0.0003,
    #             n_steps=2048,
    #             batch_size=64,
    #             n_epochs=10,
    #             gamma=0.99,
    #             gae_lambda=0.95,
    #             clip_range=0.2,
    #             clip_range_vf=None,
    #             ent_coef=0.1,
    #             vf_coef=0.5,
    #             max_grad_norm=0.5,
    #             use_sde=False,
    #             sde_sample_freq=10,
    #             target_kl=None,
    #             seed=1,
    #             policy_kwargs=policy_kwargs)
    
    model = rl_algo(CnnPolicy, env, device=device, **training_kwargs, policy_kwargs=policy_kwargs)

    output_folder_path = Path(os.getcwd()) / "output" / "test"
    tensorboard_dir = (output_folder_path / "tensorboard")

    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.tensorboard_log = tensorboard_dir.as_posix()

    checkpoint_callback = CheckpointCallback(save_freq=10000, verbose=2, save_path=ckpt_dir.as_posix())
    logging_callback = EveryNTimesteps(n_steps=100, callback=LoggingCallback(model=model))
    # eval_callback = EvalCallback(env, log_path=(output_folder_path / "eval"), eval_freq=1000)
    callbacks = CallbackList([logging_callback, checkpoint_callback])


    policy_kwargs["features_extractor_class"] = policy_kwargs["features_extractor_class"].__name__
    # writing the config information
    with open(output_folder_path / "config.json", "w") as f:
        obj = {
            "env_kwargs": env_kwargs,
            "policy_kwargs": policy_kwargs,
            "training_kwargs": training_kwargs,
            "algo": str(rl_algo.__name__)
        }
        json.dump(obj, f, indent=4)

    model = model.learn(total_timesteps=int(10), callback=callbacks, reset_num_timesteps=False)
    model.save(output_folder_path / "test")
