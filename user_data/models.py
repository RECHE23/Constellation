# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
from poutyne.framework.callbacks import lr_scheduler

ActivationFunction = nn.ReLU # Rectified Linear Unit (ReLU) activation function


# A 2 neurons per layer neural network:
class Net_2N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 2),
                                    ActivationFunction(),
                                    nn.Linear(2, 2),
                                    ActivationFunction(),
                                    nn.Linear(2, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


# A 4 neurons per layer neural network:
class Net_4N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 4),
                                    ActivationFunction(),
                                    nn.Linear(4, 4),
                                    ActivationFunction(),
                                    nn.Linear(4, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


# A 8 neurons per layer neural network:
class Net_8N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 8),
                                    ActivationFunction(),
                                    nn.Linear(8, 8),
                                    ActivationFunction(),
                                    nn.Linear(8, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


# A 16 neurons per layer neural network:
class Net_16N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 16),
                                    ActivationFunction(),
                                    nn.Linear(16, 16),
                                    ActivationFunction(),
                                    nn.Linear(16, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


# A 32 neurons per layer neural network:
class Net_32N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 32),
                                    ActivationFunction(),
                                    nn.Linear(32, 32),
                                    ActivationFunction(),
                                    nn.Linear(32, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


# A 64 neurons per layer neural network:
class Net_64N(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(dataset.data.shape[1], 64),
                                    ActivationFunction(),
                                    nn.Linear(64, 64),
                                    ActivationFunction(),
                                    nn.Linear(64, 1))

    def forward(self, input):
        return torch.erf(self.layers(input))


models = {'2 Neurons per layer': {'network': Net_2N,
                                  'optimizer': (optim.SGD, {'lr': 0.1}),
                                  'loss function': nn.MSELoss(),
                                  'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})},
          '4 Neurons per layer': {'network': Net_4N,
                                  'optimizer': (optim.SGD, {'lr': 0.1}),
                                  'loss function': nn.MSELoss(),
                                  'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})},
          '8 Neurons per layer': {'network': Net_8N,
                                  'optimizer': (optim.SGD, {'lr': 0.1}),
                                  'loss function': nn.MSELoss(),
                                  'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})},
          '16 Neurons per layer': {'network': Net_16N,
                                   'optimizer': (optim.SGD, {'lr': 0.1}),
                                   'loss function': nn.MSELoss(),
                                   'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})},
          '32 Neurons per layer': {'network': Net_32N,
                                   'optimizer': (optim.SGD, {'lr': 0.1}),
                                   'loss function': nn.MSELoss(),
                                   'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})},
          '64 Neurons per layer': {'network': Net_64N,
                                   'optimizer': (optim.SGD, {'lr': 0.1}),
                                   'loss function': nn.MSELoss(),
                                   'scheduler': (lr_scheduler.StepLR,
                                                {'step_size': 3,
                                                 'gamma': 0.99})}}
