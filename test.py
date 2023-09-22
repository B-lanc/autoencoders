import lightning as L
import torch

from model.AutoEncoder import SimpleAutoencoder
from utils import tensor_save_image, path_to_tensor

import os
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    image_path = "/saves/testing/image_00001.jpg"
    image = path_to_tensor(image_path, device=cfg.general.device)

    save_dir = os.path.join(cfg.general.checkpoint_dir, cfg.general.training_tag)
    checkpoint_path = os.path.join(
        save_dir, "lightning_logs", "version_29", "checkpoints", "checkpointepoch=00.ckpt"
    )

    MODEL = SimpleAutoencoder
    model = MODEL.load_from_checkpoint(checkpoint_path, settings=cfg.model).to(
        cfg.general.device
    )

    recreated, latents = model(image)

    tensor_save_image("/saves/testing/orig.png", image)
    tensor_save_image("/saves/testing/recc.png", recreated)


if __name__ == "__main__":
    main()
