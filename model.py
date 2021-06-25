import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

"""
this python file lists out some useful models (e.g. CNN network) and functions 
that can be used in the training. 
"""

"""
the CNN network that interleaves convolution & maxpooling layers, used in a 
previous DQN implementation and shows reasonable results
"""
class CustomMaxPoolCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMaxPoolCNN, self).__init__(observation_space, features_dim)
        # We assume CxWxH images (channels last)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=(7, 7), stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(6, 12, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
