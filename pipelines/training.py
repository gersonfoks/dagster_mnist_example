import torch
from dagster import op
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import Dataset, DataLoader

from models.feed_forward_network import LightningDnn


@op()
def train_model(model: LightningDnn, train_dataset: Dataset, val_dataset: Dataset) -> LightningDnn:
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    # Train the model ⚡
    trainer.fit(model, train_loader, val_dataloader)

    return model



@op()
def test_model(model: LightningDnn, test_dataset: Dataset) -> None:
    test_loader = DataLoader(test_dataset, batch_size=32)


    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )


    # Evaluate the model
    with torch.no_grad():
        test_results = trainer.test(model, test_loader)

    print(test_results)


