import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from dataloader import VoxDataset, VoxDataloader
from network_superclass import SoftmaxNet


class VGGnet(SoftmaxNet):

    def __init__(self, num_classes=4, lr=1e-3, batch_norm=True, dropout=0.5, L2=0.):
        SoftmaxNet.__init__(self, lr=lr, L2=L2)
        super(VGGnet, self).__init__()

        self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout
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
    tb_logger = pl_loggers.TensorBoardLogger('../Logs/', name="TestRun")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=20, tpu_cores=None, gpus=None)
    trainer.fit(model, dataloader)
