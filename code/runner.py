import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

def run(config, datapath, Dataloader, Model, logpath,
  tpu_cores=None, gpu_cores=None, phase_map_file='phase_map.csv'):
    '''

    Wrapper to run a model from a config file, output to torch lightning logs in systematic fashion
    Args:
        config: config dict for model run
        datapath: path to dataset
        Dataloader: VoxDataloader class or similar
        Model: Net model. eg: VGGnet
        logpath: path to output parent log directory

    '''

    logname = "opt{}_lr{}_reg{}_drop{}_bn{}_gc{}_mom{}".format(
        *[config[s] for s in ['optimizer', 'lr', 'L2', 'dropout', 'batch_norm', 'gradient_clipping', 'momentum']])
    print("LOGNAME: {}".format(logname))
    dataloader = Dataloader(datapath, batch_size=config['batch_size'], phase_map_file=phase_map_file)
    model = Model(num_classes=dataloader.num_classes(), lr=config['lr'], batch_norm=config['batch_norm'],
                  optimizer=config['optimizer'], momentum=config['momentum'])
    tb_logger = pl_loggers.TensorBoardLogger(logpath, name=logname)
    trainer = pl.Trainer(logger=tb_logger, max_epochs=config['max_epochs'], tpu_cores=tpu_cores, gpus=gpu_cores,
                         log_every_n_steps=20,
                         callbacks=[EarlyStopping(
                             monitor='val_loss',
                             patience=config['patience'])] if config['early_stopping'] else None)

    trainer.fit(model, dataloader)
