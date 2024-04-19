import torch
from torch.utils import data
import h5py
import numpy as np
from torchvision.transforms import Compose, Normalize, Resize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC


class MIMICImageDataset(data.Dataset):
    """
    MIMIC Image Dataset

    Args:
        data (str): Path to the h5 file containing the images.
    """

    def __init__(self, img_path, clip_pretrained=True):
        super().__init__()
        self.img_dset = h5py.File(img_path, "r")["cxr"]

        normalize_fn = Normalize(
            (101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)
        )  # mean and std of the dataset
        if clip_pretrained:
            input_resolution = 224
            transform = Compose(
                [
                    normalize_fn,
                    Resize(input_resolution, interpolation=BICUBIC),
                ]
            )
            print("Interpolation Mode: ", BICUBIC)
        else:
            input_resolution = 320
            transform = Compose(
                [
                    normalize_fn,
                ]
            )

        self.transform = transform

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # numpy, (320, 320)
        img = np.expand_dims(img, axis=0)  # numpy, (1, 320, 320)
        img = np.repeat(img, 3, axis=0)  # numpy, (3, 320, 320)
        img = torch.from_numpy(img)  # tensor, (3, 320, 320)

        if self.transform:
            img = self.transform(img)  # tensor, (3, 224, 224) or (3, 320, 320)

        sample = {"img": img}  # dict, {"img": tensor, (3, 224, 224) or (3, 320, 320)}
        return sample
