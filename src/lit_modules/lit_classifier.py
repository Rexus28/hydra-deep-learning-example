import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from typing import Any


class LitClassifier(pl.LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 loss_function: torch.nn.functional = F.cross_entropy,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use forward for inference/predictions
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('valid/loss', loss, on_step=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # self.hparams available because we called self.save_hyperparameters()
        return self.hparams.optimizer(params=self.parameters())
