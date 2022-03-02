import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch.optim as optim
import numpy as np
from dataloader import VoxDataset, VoxDataloader


class SoftmaxNet(pl.LightningModule):

    def __init__(self, lr=1e-3, L2=0., optimizer='SGD'):
        super(SoftmaxNet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.L2 = L2
        self.optimizer = optimizer

    def configure_optimizers(self):
        if self.optimizer=='sgd':
            return optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.L2)
        if self.optimizer=='adam':
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)
        else:
            raise NameError("self.optimizer not configured. Invalid Value: {}".format(self.optimizer))

    def forward(self, x):
        pass

    def predict_proba(self, x):
        return nn.Softmax(self(x))

    def predict(self, x):
        return np.argmax(self.predict_proba(x))

    def _step(self, batch, batch_idx, phase):
        label, x = batch
        logits = self(x)
        logits = logits.squeeze(1)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {phase: loss.detach()}}
        self.log("{} loss".format(phase), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


class VGGnet(SoftmaxNet):

    def __init__(self, num_classes=4, lr=1e-3, batch_norm=True, dropout=0.5, L2=0.):
        SoftmaxNet.__init__(self, lr=lr, L2=L2)
        super(VGGnet, self).__init__()

        self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.L1 = L1
        self.L2 = L2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.mpool5 = nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 3))

        self.fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(1, 9))

        # self.transpose = torch.transpose(dim0=1, dim1=3)
        self.fc65 = nn.Linear(in_features=8, out_features=1)
        self.fc7 = nn.Linear(in_features=4096, out_features=1024)
        self.fc8 = nn.Linear(in_features=1024, out_features=num_classes)

        self.seq1 = [] # initialise some sequentials for regularisation
        self.seq2 = []
        self.seq3 = []
        self.seq4 = []
        self.seq5 = []
        self.seq6 = []
        self.seq7 = []

        if self.batch_norm:
            self.seq1.append(nn.BatchNorm2d(num_features=96))
            self.seq2.append(nn.BatchNorm2d(num_features=256))
            self.seq3.append(nn.BatchNorm2d(num_features=256))
            self.seq4.append(nn.BatchNorm2d(num_features=256))
            self.seq5.append(nn.BatchNorm2d(num_features=256))
            self.seq6.append(nn.BatchNorm2d(num_features=4096))
            self.seq7.append(nn.BatchNorm1d(num_features=1024))

            self.seq1.append(nn.dropout(dropout))
            self.seq2.append(nn.dropout(dropout))
            self.seq3.append(nn.dropout(dropout))
            self.seq4.append(nn.dropout(dropout))
            self.seq5.append(nn.dropout(dropout))
            self.seq6.append(nn.dropout(dropout))
            self.seq7.append(nn.dropout(dropout))


        self.seq1 = nn.Sequential(*self.seq1)
        self.seq2 = nn.Sequential(*self.seq2)
        self.seq3 = nn.Sequential(*self.seq3)
        self.seq4 = nn.Sequential(*self.seq4)
        self.seq5 = nn.Sequential(*self.seq5)
        self.seq6 = nn.Sequential(*self.seq6)
        self.seq7 = nn.Sequential(*self.seq7)

    def forward(self, x):
        x = self.mpool1(self.activation(self.seq1(self.conv1(x))))
        x = self.mpool2(self.activation(self.seq2(self.conv2(x))))
        x = self.activation(self.seq3(self.conv3(x)))
        x = self.activation(self.seq4(self.conv4(x)))
        x = self.mpool5(self.activation(self.seq5(self.conv5(x))))
        x = self.activation(self.seq6(self.fc6(x)))
        x = x.squeeze(-1)
        x = self.fc65(x)

        x = x.view(x.size(0), -1)
        x = self.seq7(self.fc7(x))
        x = self.fc8(x)
        return x


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

        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(residual)
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class ResNet34(SoftmaxNet):

    def __init__(self, num_classes=4, lr=1e-3):
        SoftmaxNet.__init__(self, lr=lr)
        super(ResNet34, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, 2)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128, 2)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256, 2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512, 2)
        )

        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512, 11)
        # grapheme_root
        self.fc2 = nn.Linear(512, 168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512, 7)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3


if __name__ == '__main__':
    train = VoxDataset('./dataset/spectrograms/', 'train')
    valid = VoxDataset('./dataset/spectrograms/', 'validation')
    test = VoxDataset('./dataset/spectrograms/', 'test')

    datasets = [train, valid, test]

    # Load dataloader
    dataloader = VoxDataloader(train, valid, test, batch_size=3)

    print(len(train), len(valid))
    # Create model
    model = VGGnet(num_classes=4, lr=1e-3)

    # give training a go
    tb_logger = pl_loggers.TensorBoardLogger('./Logs/', name="TestRun")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=20, tpu_cores=None, gpus=None)
    trainer.fit(model, dataloader)
