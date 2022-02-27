import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class VGGnet(nn.Module):

    def __init__(self, num_classes=4):
        super(VGGnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.mpool5 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        # todo: find shape out of mpool5 to match to fc7
        self.fc7 = nn.Linear(in_features=4096, out_features=1024)
        self.fc8 = nn.Linear(in_features=1024, out_features=num_classes)

        self.stack = nn.Sequential(self.conv1, nn.ReLU(), self.mpool1,
                                    self.conv2, nn.ReLU(), self.mpool2,
                                    self.conv3, nn.ReLU(),
                                    self.conv4, nn.ReLU(),
                                    self.conv5, nn.ReLU(), self.mpool5,
                                    self.fc7,
                                    self.fc8)

    def forward(self, x):
        return self.stack(x)

    def predict_probs(self,x):
        return self.softmax(self.stack(x))

    def predict(self, x):
        return np.argmax(self.predict_probs(x))
