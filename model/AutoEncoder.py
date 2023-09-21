from typing import Sequence, Union
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import lightning as L

from .modules.Codec import Encoder, Decoder


class SimpleAutoencoder(L.LightningModule):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def training_step(self, batch, batch_idx):
        latents = self.encoder(batch)
        reconstructed = self.decoder(latents)

        mu = latents.mean()
        sigma = latents.std()
        kl_loss = 1 + (sigma ** 2).log() - sigma ** 2 - mu ** 2
        kl_loss = -kl_loss
        recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)

        self.log(
            "KL_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return kl_loss + recon_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
