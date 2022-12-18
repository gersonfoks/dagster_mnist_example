import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LightningDnn(pl.LightningModule):

    def __init__(self, deep_neural_network: nn.Module, ):
        super().__init__()

        self.deep_neural_network = deep_neural_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep_neural_network(x)

    def predict_and_evaluate(self, batch: any, batch_idx: int) -> dict:
        x, y = batch

        prediction = self.forward(x.reshape(-1, 28 * 28))

        loss = F.cross_entropy(prediction, y)

        return {"loss": loss}

    def training_step(self, train_batch: any, batch_idx: int) -> dict:
        result = self.predict_and_evaluate(train_batch, batch_idx)
        self.log('train_loss', result["loss"])
        return result

    @torch.no_grad()
    def validation_step(self, train_batch: any, batch_idx: int) -> None:
        result = self.predict_and_evaluate(train_batch, batch_idx)
        self.log('validation_loss', result["loss"])

    @torch.no_grad()
    def test_step(self, train_batch: any, batch_idx: int) -> None:
        result = self.predict_and_evaluate(train_batch, batch_idx)
        self.log('test_loss', result["loss"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.deep_neural_network.parameters())
        return optimizer
