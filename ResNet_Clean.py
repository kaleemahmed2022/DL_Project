import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataloader2 import VoxDataset, VoxDataloader



class SoftmaxNet(pl.LightningModule):

    def __init__(self, lr=1e-3):
        super(SoftmaxNet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

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
        self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "test")

class ResidualBlock(SoftmaxNet):
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

class ResNet34_clean(SoftmaxNet):
    def __init__(self):
        super(ResNet34_clean,self).__init__()

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

if __name__ == '__main__':
    train = VoxDataset('./dataset/spectrograms/', 'train')
    valid = VoxDataset('./dataset/spectrograms/', 'validation')
    test = VoxDataset('./dataset/spectrograms/', 'test')

    datasets = [train, valid, test]

    # Load dataloader
    dataloader = VoxDataloader(train, valid, test)

    # Create model
    model = ResNet34_clean()

    # quick test
    pred = model.predict_proba(test[0][1].unsqueeze(0))
    print(pred)

    # give training a go
    tb_logger = pl_loggers.TensorBoardLogger('./Logs/', name="TestRun")
    trainer = pl.Trainer(logger= tb_logger, max_epochs=20, tpu_cores=None, gpus=None)
    trainer.fit(model, dataloader)
