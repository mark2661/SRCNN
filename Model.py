import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.l1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate')
        nn.init.xavier_uniform_(self.l1.weight)

        self.l2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x
