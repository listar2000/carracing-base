"""
this file keeps some of the common utility functions that will be 
used by different agents in training. e.g. stack different images,
discretize action space...
"""

from typing import Dict, List, Tuple
from collections import deque
import numpy as np
import cv2
import gym
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from config import TRAINING_CONFIG
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete


def discretize_action_space(
        steering_acs: List[float] = [],
        throttle_acs: List[float] = [],
        brake_acs: List[float] = [], **kwargs
) -> np.ndarray:
    """
    :param steering_acs: the list of values (between -1 and 1) for steering
    :param throttle_acs: the list of values (between 0 and 1) for throttle
    :param brake_acs: the list of values (between 0 and 1) for brake

    Additional params:
    :param dims:
    you might choose not to offer any of the above params, and
    instead offer a list of three numbers, e.g. [3, 3, 2], which means that
    you want to have 3x3x2=18 actions uniformly scattered in the 3D action
    space.

    :return
    a list of triples representing valid, discretized actions (essentially a
    cartesian product of the three input lists)
    """
    if not (steering_acs or throttle_acs or brake_acs):
        dims = kwargs.get('dims', [])
        if dims:
            assert np.all([type(d) == int and d > 0 for d in dims]), \
                "dims must be integers > 0"
            steering_acs = np.linspace(-1, 1, dims[0])
            throttle_acs = np.linspace(0, 1, dims[1])
            brake_acs = np.linspace(0, 1, dims[2])
        else:
            return None

    # perform cartesian product
    actions = []
    for sac in steering_acs:
        for tac in throttle_acs:
            for bac in brake_acs:
                actions.append((sac, tac, bac))

    return np.array(actions)


"""
the original state from env.step() has shape 96x96x3 where the last
dimension is in BGR format. This function converts the BGR image into
a gray-scale image with shape 96x96x1, taking values in [0, 1]
"""


def preprocess_state(original_state: np.ndarray) -> np.ndarray:
    assert original_state.shape[-1] == 3
    gray_state = cv2.cvtColor(original_state, cv2.COLOR_BGR2GRAY)
    return gray_state


"""
this function stacks several states of size n
and returns a stacked state of size 96 x 96 x n
"""


def stack_states(states: np.ndarray) -> np.ndarray:
    assert states.shape[1:] == (96, 96)
    return np.transpose(states, (1, 2, 0))


"""
A wrapper class for the carracing-v0 environment that supports frame stacking,
skipping and early-stopping (see `self.negative_reward_counter` and related codes).
"""

DEFAULT_AC_SPACE = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
]

AC_SPACE_V2 = [
    (-1, 0.5, 0.2), (0, 0.5, 0.2), (1, 0.5, 0.2), #           Action Space Structure
    (-1, 0.5,   0), (0, 0.5,   0), (1, 0.5,   0), #        (Steering Wheel, Gas, Break)
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
    (-1, 0,   0), (0, 0,   0), (1, 0,   0),
    (0.5,0.5, 0.2), (0.5,0, 0.2), (0.5,0.5, 0),
    (0.5,0,0), (-0.5,0.5, 0.2), (-0.5,0, 0.2), 
    (-0.5,0.5, 0), (-0.5,0,0)
]

ACTION_TYPES = ['CONTINUOUS', 'DISCRETE', 'MULTI_DISCRETE']

class env_wrapper(gym.Env):

    def __init__(self, env: gym.Env,
                 frame_stack_num: int,
                 skip_frame: int = TRAINING_CONFIG.get("skip_frame", 0),
                 done_threshold: int = TRAINING_CONFIG.get("done_threshold", 100),
                 action_space_type: str = "continuous",
                 num_discrete_steering: int = 10,
                 num_discrete_throttle: int = 10,
                 num_discrete_brake: int = 2,
                 action_space: List = DEFAULT_AC_SPACE,
                 is_training: bool = True
                 ) -> None:
        
        self.is_training = is_training
        self.action_space_type = action_space_type.upper()
        assert self.action_space_type in ACTION_TYPES

        self.frame_stack_num = frame_stack_num
        self.skip_frame = skip_frame
        self.done_threshold = done_threshold * 3 if not self.is_training else done_threshold

        self.env = env
        self.state_deque = None
        self.discrete_steerings = np.linspace(-1, 1, num_discrete_steering)
        self.discrete_gases = np.linspace(0, 1, num_discrete_throttle)
        self.discrete_brakes = np.linspace(0, 1, num_discrete_brake)
        if self.action_space_type == 'MULTI_DISCRETE':
            self.action_space = MultiDiscrete([len(self.discrete_steerings),
                                               len(self.discrete_gases),
                                               len(self.discrete_brakes)])
        elif self.action_space_type == 'DISCRETE':
            self.action_space = Discrete(len(action_space))
            self._action_space = action_space
        else:
            self.action_space = env.action_space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(96, 96, self.frame_stack_num), dtype=np.uint8
        )

        print("SHAPE:" + str(self.observation_space.shape))

        # for early stopping (better for training)
        self.negative_reward_counter = 0
        self.step_counter = 0
        self.episode_counter = 0
        self.total_reward = 0

        self.should_render = False
        self.every_render = None

    def discrete_to_continue_action(self, action: np.ndarray) -> List:
        if isinstance(action, np.ndarray) and len(action) == 3: # multi discrete case
            steering_index, gas_index, brake_index = action
            return [self.discrete_steerings[steering_index],
                    self.discrete_gases[gas_index],
                    self.discrete_brakes[brake_index]]
        else: # discrete case
            return self._action_space[action]

    def set_render(self, render: bool=False, every=None):
        if every is not None:
            self.every_render = every
        self.should_render = render

    # return the initial state and initialize the state deque as well
    def reset(self) -> np.ndarray:
        init_state = self.env.reset()
        init_state = preprocess_state(init_state)

        self.state_deque = deque([init_state] * self.frame_stack_num, maxlen=self.frame_stack_num)

        self.step_counter = 0
        self.negative_reward_counter = 0
        self.total_reward = 0

        return stack_states(np.array(self.state_deque))

    # as usual, return (next_state, reward, done, info)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        reward_sum = 0
        done = False
        info = dict()

        should_render = self.should_render
        if self.every_render is not None:
            should_render = self.episode_counter % self.every_render == 0

        if self.action_space_type != 'CONTINUOUS':
            action = self.discrete_to_continue_action(action=action)

        for _ in range(self.skip_frame + 1):
            next_state, reward, done, info = self.env.step(action=action)
            # update our state deque
            next_state = preprocess_state(next_state)
            self.state_deque.append(next_state)

            if should_render:
                self.env.render()

            reward_sum += reward
            if done:
                break
        
        # amplify reward for using full gas
        if self.is_training and action[1] > 0.9 and action[2] == 0:
            reward_sum *= 1.5
        
        self.total_reward += reward_sum
        next_state = stack_states(np.array(self.state_deque))

        # update the counter
        self.step_counter += (self.skip_frame + 1)

        if self.step_counter < 100 or reward_sum > 0:
            self.negative_reward_counter = 0
        else:
            self.negative_reward_counter += 1

        # the total reward penalty is not applied during testing
        if self.negative_reward_counter > self.done_threshold or (self.total_reward < 0 and self.is_training):
            info["done by early stopping"] = True
            done = True

        info["reward sum"] = reward_sum
        if self.action_space_type != 'CONTINUOUS':
            info["parsed action"] = action

        if done:
            self.episode_counter += 1

        return next_state, reward_sum, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)


class LoggingCallback(BaseCallback):
    def __init__(self, model: BaseAlgorithm, verbose=0):
        super().__init__(verbose)
        self.init_callback(model=model)

    # def _on_step(self) -> bool:
    #     print(
    #         f"""
    #         Step = {self.num_timesteps}
    #         Current Reward = {self.locals['rewards']}
    #         Current Actions = {self.locals['actions']} -> {self.locals['clipped_actions']}
    #         debug info = {self.locals['infos']}
    #         """
    #     )
    #     return True
    
    def _on_step(self) -> bool:
        print(
            f"""
            Step = {self.num_timesteps}
            Current Reward = {self.locals['reward']}
            Current Actions = {self.locals['action']}
            debug info = {self.locals['infos']}
            """
        )
        return True
