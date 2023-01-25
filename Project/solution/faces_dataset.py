"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple([torch.Tensor, int]):
        """Get a sample and label from the dataset."""
        if index < len(self.real_image_names):
            folder = 'real'
            img_name = self.real_image_names[index]
            fake = 0
        else :
            folder = 'fake'
            img_name = self.fake_image_names[index-len(self.real_image_names)]
            fake = 1

        img_path = os.path.join(self.root_path, folder, img_name)
        with Image.open(img_path) as im:
            if self.transform is not None:
                smp = self.transform(im)
        return smp, fake


    def __len__(self):
        """Return the number of images in the dataset."""
        return (len(self.real_image_names)+len(self.fake_image_names))

