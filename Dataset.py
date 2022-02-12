import torch
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):

    def __init__(self, images, labels):
        """labels is also a numpy ndarray representing an image"""
        """
        Not using transforms.ToTorch()in __getitem__ becuase the input from the h5 file is already in the form [C, H, W]
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(image, dtype=torch.float)
        )