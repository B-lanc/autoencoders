import torch
from PIL import Image
import albumentations as A
import numpy as np


def path_to_tensor(path, device="cuda", resize=True, crop=True):
    transforms = []
    if resize:
        transforms.append(A.SmallestMaxSize(256))
    if crop:
        transforms.append(A.CenterCrop(height=256, width=256))

    image = np.array(Image.open(path).convert("RGB")).astype(np.uint8)

    if len(transforms) > 0:
        transform = A.Compose(transforms)
        image = transform(image=image)["image"]

    image = image / 127.5 - 1
    image = image.transpose(2, 0, 1)[None, :, :, :]
    image = torch.Tensor(image)
    image = image.to(device)
    return image


def tensor_save_image(path, tensor):
    image = tensor.detach().cpu().numpy()[0, :, :, :]
    image = ((image + 1) * 127.5).astype(np.uint8)
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)
    image.save(path)
