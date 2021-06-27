from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import CnnPolicy
# from stable_baselines3.td3 import TD3
# from stable_baselines3.td3.policies import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, EveryNTimesteps, CallbackList
from utils import LoggingCallback, env_wrapper
from model import CustomMaxPoolCNN
from pathlib import Path
from carracing import CarRacing
import torch as th
import os
from datetime import datetime

if __name__ == '__main__':
    wrapped_env = CarRacing(diff_road_col=True)
    env = env_wrapper(env=wrapped_env,
                      frame_stack_num=4,
                      skip_frame=2,
                      use_discrete_action_space=True)

    env.set_render(True)

    if th.cuda.is_available():
        print("using cuda:0 right now...")
        device = th.device('cuda:0')
    else:
        print("no cuda available, cpu used for training...")
        device = th.device('cpu')

    policy_kwargs = dict(
        features_extractor_class=CustomMaxPoolCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )

    model = PPO(CnnPolicy, env,
                device=device,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.1,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=10,
                target_kl=None,
                seed=1,
                policy_kwargs=policy_kwargs)

    output_folder_path = Path(os.getcwd()) / "output" / "ppo"
    tensorboard_dir = (output_folder_path / "tensorboard")

    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.tensorboard_log = tensorboard_dir.as_posix()

    checkpoint_callback = CheckpointCallback(save_freq=1000, verbose=2, save_path=ckpt_dir.as_posix())
    logging_callback = EveryNTimesteps(n_steps=1, callback=LoggingCallback(model=model))
    # eval_callback = EvalCallback(env, log_path=(output_folder_path / "eval"), eval_freq=1000)
    callbacks = CallbackList([logging_callback, checkpoint_callback])

    model = model.learn(total_timesteps=int(1e6), callback=callbacks, reset_num_timesteps=False)
    model.save(output_folder_path / "carracing_v0_ppo")
