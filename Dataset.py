import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        super(TrainingDataset, self).__init__()
        self.h5_file = h5_file
        self.transform = transform

    def __getitem__(self, idx):
        """
        Not using transforms.ToTorch() because the input from the h5 file is already in the form [C, H, W],
        instead using torch.tensor which converts a numpy array without re-ordering dimensions then passing the
        torch tensor into the composite transform function for random flips and rotations.
        """
        with h5py.File(self.h5_file, 'r') as f:
            image = torch.tensor(np.expand_dims(f['lr'][idx] / 255., 0), dtype=torch.float)
            label = torch.tensor(np.expand_dims(f['hr'][idx] / 255., 0), dtype=torch.float)

        if self.transform:
            return self.transform(image), self.transform(label)

        return image, label

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            size = len(f['lr'])
        return size


class ValidationDataset(Dataset):
    def __init__(self, h5_file):
        super(ValidationDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        """
        Not using transforms.ToTorch() because the input from the h5 file is already in the form [C, H, W]
        """
        with h5py.File(self.h5_file, 'r') as f:
            image = np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0)
            label = np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            size = len(f['lr'])
        return size
