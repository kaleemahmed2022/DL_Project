import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from dataloader import VoxDataset, VoxDataloader
from network_superclass import SoftmaxNet


class VGGmini(SoftmaxNet):

    def __init__(self, num_classes=20, lr=1e-3, batch_norm=True, dropout=0.5, L2=0., optimizer='sgd', momentum=0.):
        '''
        aprox 4x smaller than the full blown VGGnet and only 3 conv layers
        '''

        SoftmaxNet.__init__(self, lr=lr, L2=L2, optimizer=optimizer, momentum=momentum)
        super(VGGmini, self).__init__()

        self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.L2 = L2
        self.optimizer = optimizer
        self.momentum = momentum 

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=1024, kernel_size=5, stride=2, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        #self.fc3 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=(1, 9))
        self.fc3 = nn.Linear(in_features=31, out_features=1)

        # self.transpose = torch.transpose(dim0=1, dim1=3)
        self.fc4 = nn.Linear(in_features=17, out_features=1)
        self.fc5 = nn.Linear(in_features=1024, out_features=num_classes)

        self.seq1 = [] # initialise some sequentials for regularisation
        self.seq2 = []
        self.seq3 = []
        self.seq5 = []

        if self.batch_norm:
            self.seq1.append(nn.BatchNorm2d(num_features=96))
            self.seq2.append(nn.BatchNorm2d(num_features=1024))
            self.seq3.append(nn.BatchNorm2d(num_features=1024))

            self.seq1.append(nn.Dropout(dropout))
            self.seq2.append(nn.Dropout(dropout))
            self.seq3.append(nn.Dropout(dropout))

        self.seq1 = nn.Sequential(*self.seq1)
        self.seq2 = nn.Sequential(*self.seq2)
        self.seq3 = nn.Sequential(*self.seq3)

    def forward(self, x):
        x = self.mpool1(self.activation(self.seq1(self.conv1(x))))
        x = self.mpool2(self.activation(self.seq2(self.conv2(x))))
        #x = self.mpool3(self.activation(self.seq3(self.fc3(x))))
        x = self.activation(self.seq3(self.fc3(x)))
        x = x.squeeze(-1)
        x = self.fc4(x)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return x


if __name__ == '__main__':
    # Load dataloader
    #dataloader = VoxDataloader('../dataset/raw/', batch_size=3)
    dataloader = VoxDataloader('/Users/jameswilkinson/Downloads/dev/wav3/', batch_size=32)

    # Create model
    model = VGGmini(num_classes=4, lr=1e-3)

    # give training a go
    tb_logger = pl_loggers.TensorBoardLogger('../Logs/', name="TestRun")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=20, tpu_cores=None, gpus=None)
    trainer.fit(model, dataloader)
