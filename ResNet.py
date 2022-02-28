import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pytorch_lightning as pl
import torchmetrics
import torch
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.datasets as datasets
import pytorch_lightning.loggers as pl_loggers
from dataloader2 import VoxDataset


# # 3x3 convolution
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'train': accuracy.detach()}, 'loss': {'train': loss.detach()}}
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/train", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    # def training_epoch_end(self, outputs):
    #     sampleImg=torch.rand((1,1,28,28))
    #     self.logger.experiment.add_graph(MNISTLightningModule(), sampleImg)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'val': accuracy.detach()}, 'loss': {'val': loss.detach()}}
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/val", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'test': accuracy.detach()}, 'loss': {'test': loss.detach()}}
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/test", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

# ResNet 34 example attempt
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )

        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'train': accuracy.detach()}, 'loss': {'train': loss.detach()}}
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/train", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    # def training_epoch_end(self, outputs):
    #     sampleImg=torch.rand((1,1,28,28))
    #     self.logger.experiment.add_graph(MNISTLightningModule(), sampleImg)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'val': accuracy.detach()}, 'loss': {'val': loss.detach()}}
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/val", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = torchmetrics.functional.accuracy(logits, y)
        tensorboard_logs = {'acc': {'test': accuracy.detach()}, 'loss': {'test': loss.detach()}}
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/test", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}


# class ResNet34_DateModule(pl.LightningDataModule):
#     def __init__(self, data_dir='./dataset/processed/'):
#         super().__init__()
#         self.data_dir = data_dir
#         self.transform = transforms.Compose([transforms.ToTensor()])
#
#     def setup(self, stage=None):
#         self.res_train = datasets.VoxDataset('./dataset/processed/', train=True)
#         self.res_test = datasets.VoxDataset('./dataset/processed/', train=False)
#
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.res_train, batch_size=32, shuffle=True)
#
#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.res_test, batch_size=32, shuffle=False)


model = ResNet34()
data_module = VoxDataloader(train, valid, test)
#  dataloader = VoxDataloader(train, valid, test)
# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html#tensorboard-logger

# You can change the name attribute to your question number.
# This will update in your graph legends and will make reading your graphs easier.
tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="ResNet34")

trainer = pl.Trainer(logger=tb_logger, max_epochs=10)
trainer.fit(model, data_module)

result = trainer.test(model, data_module)
print(result)
