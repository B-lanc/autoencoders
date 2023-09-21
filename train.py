import lightning as L
import torch
from torch.utils.data import DataLoader

from model.AutoEncoder import SimpleAutoencoder
from dataset import dataset

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
    ds = dataset(cfg.general.dataset_dir, cfg.model.resolution)
    dataloader = DataLoader(
        ds, batch_size=cfg.general.batch_size, shuffle=True, num_workers=6
    )

    trainer = L.Trainer(
        accelerator="gpu", max_epochs=500, min_epochs=50, default_root_dir=save_dir
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
