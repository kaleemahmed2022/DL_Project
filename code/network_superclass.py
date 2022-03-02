import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np


class SoftmaxNet(pl.LightningModule):

    __doc__ = '''
    Lighning module to harbour the basic functionality for all
    neural networks that use a softmax output
    '''

    def __init__(self, lr=1e-3, L2=0., optimizer='SGD'):
        super(SoftmaxNet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.L2 = L2
        self.optimizer = optimizer.lower()

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
        self.log("{}_loss".format(phase), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
