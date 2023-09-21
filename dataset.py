from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np

import os


class dataset(Dataset):
    """stolen from outlier"""

    def __init__(self, path, resolution=256):
        self.data = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.data)

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=resolution),
                A.RandomCrop(width=resolution, height=resolution),
                A.HorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.preprocess(self.data[index])

    def preprocess(self, path):
        im = np.array(Image.open(path).convert("RGB"))
        im = self.transform(image=im)["image"]
        im = (im / 127.5 - 1.0).astype(np.float32).transpose(2, 0, 1)
        return im
