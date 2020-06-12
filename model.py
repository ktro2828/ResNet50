import torch
import torch.nn as nn
import torch.nn.function as F
import torch.optim as optimizers
import torchvision.transforms as transforms

class Resnet50(nn.Module):
    def __init__(self, output_dim):
        super(Resnet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        # Block1
        self.block0 = self._build(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._build(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))

        # Block2
        self.block2 = nn.ModuleList([
            self._build(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))

        # Block4
        self.block4 = nn.Module([
            self._build(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()
        self.fc1 = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(self.bn1(h), inplace=True)
        h = self.pool1(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=1)

        return y

    def _build(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)

class Block(nn.Module):
    def __init__(self, channel_in, channel_out, drop_rate=0.3):
        super().__init__()

        channel = channel_out

        self.bn1 = nn.BatchNorm2d(channel_in)
        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(3, 3))
        self.drop_rate = drop_rate
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.shortcut = self._shortcut(channel_in, channel_out)

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=True)
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = F.dropout(h, p=self.drop_rate)
        h = self.conv2(h)
        y = F.relu(h + self.shortcut(x))

        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return labmda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out,kernel_size=(1, 1), padding=0)

    class GlobalAvgPool2d(nn.Module):
        def __init__(self, device='cpu'):
            super().__init__

        def forward(self, x):
            return F.avg_pool2d(x, kernel_size=x.size()[2:].view(01, x.size(1)))
