### Here we define the ops for creating a new model.
from pathlib import Path

import yaml
from yaml.loader import SafeLoader
from dagster import op, graph, GraphOut

from models.feed_forward_network import LightningDnn
from models.model_factory import FeedForwardNetworkFactory
from models.model_manager import LightningDnnManager


@op(config_schema={"model_config_location": str})
def load_config(context: any) -> dict:
    '''
    Load the model config.
    This config defines the model architecture as well as the location to save the model.
    Lastly it also defines the training parameters.
    :param context:
    :return:
    '''
    model_config_location = context.op_config["model_config_location"]
    with open(model_config_location, 'r') as f:
        model_config = yaml.load(f, Loader=SafeLoader)

    return model_config


@op()
def create_model_from_config(model_config: dict) -> LightningDnn:
    factory = FeedForwardNetworkFactory()
    model = factory.build_from_config(model_config)
    return model


@graph(out={"model": GraphOut(), "model_config": GraphOut(), })
def create_model() -> (LightningDnn, dict):
    model_config = load_config()
    model = create_model_from_config(model_config)
    return model, model_config


@op()
def save_model(model: LightningDnn, model_config: dict) -> str:
    '''
    Save the model to the specified location
    :param model: lightning model
    :param model_config: model config
    :return: None
    '''

    model_manager = LightningDnnManager()

    model_manager.save_model(model, model_config)

    return model_config["save_location"]


@op()
def load_model(save_location: str) -> LightningDnn:
    '''
    Load the model from the specified location
    :param model_save_location: location to save the model
    :return: lightning model
    '''

    model_manager = LightningDnnManager()

    return model_manager.load_model(Path(save_location))
