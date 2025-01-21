import torch
import torch.nn as nn
from torchvision.models import resnet18

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class ResNet18(BaseFeaturesExtractor):
    """
    A custom feature extractor based on ResNet 18
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, weights=None):
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]

        # load resnet-18 model
        self.resnet = resnet18(weights=weights)

        # modify the first conv layer to accept channels according to the environment
        self.resnet.conv1 = nn.Conv2d(
            in_channels=n_input_channels,  # Set the new input channels
            out_channels=self.resnet.conv1.out_channels,  # Keep the same number of output channels
            kernel_size=self.resnet.conv1.kernel_size,  # Keep kernel size
            stride=self.resnet.conv1.stride,  # Keep stride
            padding=self.resnet.conv1.padding,  # Keep padding
            bias=self.resnet.conv1.bias is not None,  # Match bias usage
        )

        # remove the last fully connected layer, which was designed for classification
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return x
