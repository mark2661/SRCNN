import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.l1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.l2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.l3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x), inplace=True)
        x = nn.functional.relu(self.l2(x), inplace=True)
        x = self.l3(x)
        return x
