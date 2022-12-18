"""
Makes use of the factory method to create new neural networks
"""
from abc import abstractmethod

import pytorch_lightning as pl
from torch import nn

from models.feed_forward_network import LightningDnn


class AbstractFactory:

    @abstractmethod
    def build_from_config(self, config: dict) -> pl.LightningModule:
        pass


class FeedForwardNetworkFactory(AbstractFactory):
    '''
        Simple factory method for creating a new Feed Forward neural network
    '''

    def build_from_config(self, config: dict) -> LightningDnn:
        layers = []
        for layer in config["layers"][:-1]:
            layers.append(nn.Linear(layer["input_size"], layer["output_size"]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(config["layers"][-1]["input_size"], config["layers"][-1]["output_size"]))

        feed_forward_network = nn.Sequential(*layers)

        return LightningDnn(feed_forward_network)
