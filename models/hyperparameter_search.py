import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import Dataset, DataLoader

from models.feed_forward_network import LightningDnn
from models.model_factory import FeedForwardNetworkFactory

network_size_dicts = {
    "small": {
        "layers": [
            {"input_size": 28 * 28, "output_size": 128},
            {"input_size": 128, "output_size": 10},
        ]
    },
    "medium": {
        "layers": [
            {"input_size": 28 * 28, "output_size": 256},
            {"input_size": 256, "output_size": 128},
            {"input_size": 128, "output_size": 10},
        ]
    },
    "large": {
        "layers": [
            {"input_size": 28 * 28, "output_size": 512},
            {"input_size": 512, "output_size": 256},
            {"input_size": 256, "output_size": 128},
            {"input_size": 128, "output_size": 10},
        ]
    },
}


class HyperparameterSearch:

    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_trails = 1

        self.best_model = None
        self.best_val_loss = None


    def search(self):
        '''
        Perform hyperparameter search
        :return:
        '''

        study = optuna.create_study(direction="minimize")

        study.optimize(self.objective, n_trials=self.n_trails)

        best_model = self.best_model



        return study, best_model

    def objective(self, trial) -> float:
        '''
        Objective function to optimize
        :param trial:
        :return:
        '''

        hyperparameters = self.get_hyperparameters(trial)

        model = self.get_model(hyperparameters["model_config"])
        trainer = self.get_trainer(hyperparameters["trainer_config"])

        train_loader = DataLoader(self.train_dataset, batch_size=hyperparameters["batch_size"])
        val_dataloader = DataLoader(self.val_dataset, batch_size=hyperparameters["batch_size"])
        trainer.fit(model, train_loader, val_dataloader)

        val_loss = trainer.validate(model, val_dataloader)[0]["validation_loss"]

        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = model


        return val_loss

    def get_hyperparameters(self, trial) -> dict:
        '''
        Define the search space
        :return:
        '''

        model_config = self.get_model_hyperparameters(trial)

        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, ])

        trainer_config = {
            "max_epochs": trial.suggest_int("max_epochs", 1, 10),

        }

        return {"model_config": model_config, "trainer_config": trainer_config, "batch_size": batch_size}

    def get_model_hyperparameters(self, trial) -> dict:
        '''
        Get model hyperparameters
        :param trial:
        :return:
        '''

        # First suggest the size of the network:
        network_size = trial.suggest_categorical("network_size", ["small", "medium", "large"])

        # define the layer sizes based on the network size:
        layers = network_size_dicts[network_size]["layers"]

        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

        return {
            "layers": layers,
            "learning_rate": learning_rate,
        }

    def get_model(self, model_config: dict) -> LightningDnn:
        '''
        Get model
        :return:
        '''

        model = FeedForwardNetworkFactory().build_from_config(model_config)

        return model

    def get_trainer(self, trainer_config: dict) -> any:
        '''
        Get trainer
        :return:
        '''

        # Initialize a trainer
        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=trainer_config["max_epochs"],

            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )

        return trainer
