import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from model.AutoEncoder import SimpleAutoencoder
from dataset import make_train_val

import os
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    tag = cfg.general.training_tag

    save_dir = os.path.join(cfg.general.checkpoint_dir, tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    MODEL = SimpleAutoencoder

    model = MODEL(settings=cfg.model)
    train_ds, val_ds = make_train_val(cfg.general.dataset_dir, cfg.model.resolution)
    train_dataloader = DataLoader(
        train_ds, batch_size=cfg.general.batch_size, shuffle=True, num_workers=6
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=cfg.general.batch_size, num_workers=6
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename="checkpoint{epoch:02d}",
    )
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=100,
        min_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
