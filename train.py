import lightning as L
import torch
from torch.utils.data import DataLoader

from model.AutoEncoder import SimpleAutoencoder
from dataset import dataset
import settings

import os


def main():
    tag = settings.training_tag
    save_dir = os.path.join(settings.checkpoint_dir, tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    MODEL = SimpleAutoencoder

    model = MODEL()
    ds = dataset(settings.dataset_dir, settings.resolution)
    dataloader = DataLoader(
        ds, batch_size=settings.batch_size, shuffle=True, num_workers=6
    )

    trainer = L.Trainer(
        accelerator="gpu", max_epochs=300, min_epochs=50, default_root_dir=save_dir
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
