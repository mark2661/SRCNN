import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, first_layer_filter_num):
        super(SRCNN, self).__init__()
        self.first_layer_filter_num = first_layer_filter_num
        self.l1 = nn.Conv2d(1, self.first_layer_filter_num, kernel_size=9, padding=9 // 2)
        self.l2 = nn.Conv2d(self.first_layer_filter_num, self.first_layer_filter_num // 2, kernel_size=5, padding=5 // 2)
        self.l3 = nn.Conv2d(self.first_layer_filter_num // 2, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x), inplace=True)
        x = nn.functional.relu(self.l2(x), inplace=True)
        x = self.l3(x)
        return x
