import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from dataloader import VoxDataset, VoxDataloader
from network_superclass import SoftmaxNet


class VGGmini(SoftmaxNet):

    def __init__(self, num_classes=20, lr=1e-3, batch_norm=True, dropout=0.5, L2=0., momentum=0.,lr_decay=0.,optimizer='sgd'):
        '''
        aprox 4x smaller than the full blown VGGnet and only 3 conv layers
        '''
        super(VGGmini, self).__init__(lr=lr, L2=L2, momentum=momentum, lr_decay = lr_decay, optimizer=optimizer)

        self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=1024, kernel_size=5, stride=2, padding=1)

        #self.fc3 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=(1, 9))
        self.fc3 = nn.Linear(in_features=31, out_features=1)

        # self.transpose = torch.transpose(dim0=1, dim1=3)
        self.fc4 = nn.Linear(in_features=512, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_classes)


        # self.seq1 = [] # initialise some sequentials for regularisation
        # self.seq2 = []
        # self.seq3 = []
        # self.seq5 = []

        # if self.batch_norm:
        #     self.seq1.append(nn.BatchNorm2d(num_features=96))
        #     self.seq2.append(nn.BatchNorm2d(num_features=1024))
        #     self.seq3.append(nn.BatchNorm2d(num_features=1024))

        #     self.seq1.append(nn.Dropout(dropout))
        #     self.seq2.append(nn.Dropout(dropout))
        #     self.seq3.append(nn.Dropout(dropout))

        # self.seq1 = nn.Sequential(*self.seq1)
        # self.seq2 = nn.Sequential(*self.seq2)
        # self.seq3 = nn.Sequential(*self.seq3)

    # def forward(self, x):
    #     x = self.activation(self.seq1(self.conv1(x)))
    #     x = self.activation(self.seq2(self.conv2(x)))
    #     #x = self.mpool3(self.activation(self.seq3(self.fc3(x))))
    #     x = self.activation(self.seq3(self.fc3(x)))
    #     x = x.squeeze(-1)
    #     x = self.fc4(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc5(x)

    #     return x
    def forward(self,x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.mpool1(x)
        x = self.activation(self.conv3(x))
        x = self.mpool2(x)
        x = nn.Flatten()(x)
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))

        return x

if __name__ == '__main__':
    # Load dataloader
    #dataloader = VoxDataloader('../dataset/raw/', batch_size=3)
    dataloader = VoxDataloader('/Users/jameswilkinson/Downloads/dev/wav3/', batch_size=32, phase_map_file='phase_map_small.csv',
                               fftmethod='librosa.mel')
    print("Num classes: {}".format(dataloader.num_classes()))

    # Create model
    model = VGGmini(num_classes=dataloader.num_classes(), lr=1e-3, optimizer='SGD')

    # give training a go
    tb_logger = pl_loggers.TensorBoardLogger('../Logs/', name="TestRun")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=20, tpu_cores=None, gpus=None)
    trainer.fit(model, dataloader)
