from typing import Any, Optional, Sequence, Union
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import lightning as L

from .modules.Codec import Encoder, Decoder


class SimpleAutoencoder(L.LightningModule):
    def __init__(self, settings):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Encoder(
            settings.img_channel,
            settings.base_channel,
            settings.ch_mult,
            settings.z_channel,
            settings.resolution,
            settings.depth,
            settings.attention_res,
            settings.dropout,
        )
        self.decoder = Decoder(
            settings.img_channel,
            settings.base_channel,
            settings.ch_mult,
            settings.z_channel,
            settings.resolution,
            settings.depth,
            settings.attention_res,
            settings.dropout,
        )
        self.kl_const = settings.kl_const
        self.lr = settings.lr

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def training_step(self, batch, batch_idx):
        latents = self.encoder(batch)
        reconstructed = self.decoder(latents)

        mu = latents.mean()
        sigma = latents.std()
        kl_loss = 1 + (sigma ** 2).log() - sigma ** 2 - mu ** 2
        kl_loss = -kl_loss * self.kl_const
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

    def validation_step(self, batch, batch_idx):
        preds, _ = self(batch)
        loss = torch.nn.functional.mse_loss(preds, batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
