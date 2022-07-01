import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        """
        currently network is of the form:
                                        first row = 256 filters
                                        second row = 128 filters
                                        third layer = 1 filter
        This can be altered if required.
        padding uses formula taken from https://www.sciencedirect.com/science/article/abs/pii/S0925231219312974
        Kernel sizes follows model outlined in https://ieeexplore.ieee.org/abstract/document/7115171
        """
        super(SRCNN, self).__init__()
        self.l1 = nn.Conv2d(1, 256, kernel_size=9, padding=9 // 2)
        self.l2 = nn.Conv2d(256, 128, kernel_size=5, padding=5 // 2)
        self.l3 = nn.Conv2d(128, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x), inplace=True)
        x = nn.functional.relu(self.l2(x), inplace=True)
        x = self.l3(x)
        return x
