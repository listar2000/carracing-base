from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import CnnPolicy
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
    env = env_wrapper(env=wrapped_env, frame_stack_num=3, skip_frame=2)

    env.set_render(True)

    if th.cuda.is_available():
        print("using cuda:0 right now...")
        device = th.device('cuda:0')
    else:
        print("no cuda available, cpu used for training...")
        device = th.device('cpu')

    policy_kwargs = dict(
        features_extractor_class=CustomMaxPoolCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = TD3(CnnPolicy, env, buffer_size=5000, device=device, policy_kwargs=policy_kwargs)
    
    output_folder_path=Path(os.getcwd()) / "output" / "td3-2"
    tensorboard_dir = (output_folder_path / "tensorboard")
    
    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.tensorboard_log = tensorboard_dir.as_posix()
    # checkpoint_callback = CheckpointCallback(save_freq=10000, verbose=2, save_path=ckpt_dir.as_posix())

    logging_callback = EveryNTimesteps(n_steps=1000, callback=LoggingCallback(model=model))
    eval_callback = EvalCallback(env, log_path=(output_folder_path / "eval"), eval_freq=1000)
    callbacks = CallbackList([logging_callback])
    model = model.learn(total_timesteps=int(1e5), callback=callbacks, reset_num_timesteps=False)
    model.save(output_folder_path / "carracing_v0_td3_2")

