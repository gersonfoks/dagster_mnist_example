from pathlib import Path

import torch

from models.feed_forward_network import LightningDnn
from models.model_factory import FeedForwardNetworkFactory


class LightningDnnManager:
    '''
        This class is responsible for loading and saving models.
    '''


    def save_model(self, model: LightningDnn, model_config: dict) -> None:
        '''
        Save the model to the specified location
        :param model: lightning model
        :return: None
        '''

        model_save_location = Path(model_config["save_location"])

        model_save_location.mkdir(parents=True, exist_ok=True)
        pl_path = model_save_location / 'pl_model.pt'

        state = {
            "config": model_config,
            "state_dict": model.deep_neural_network.state_dict()
        }

        torch.save(state, pl_path)

    def load_model(self, model_save_location: Path) -> LightningDnn:
        '''
        Load the model from the specified location
        :return: lightning model
        '''

        pl_path = model_save_location / 'pl_model.pt'
        checkpoint = torch.load(pl_path)
        model = FeedForwardNetworkFactory().build_from_config(checkpoint["config"])
        model.deep_neural_network.load_state_dict(checkpoint["state_dict"])
        return model
