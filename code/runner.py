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
        tpu_cores: number of tpu cores to use
        gpu_cores: number of gpu cores to use
        phase_map_file: name of the phase_map that identifies files to training/valid/test phases
    '''

    # log set up
    logname = "spct{}_opt{}_lr{}_reg{}_drop{}_bn{}_gc{}_mom{}_dec{}_pooling{}".format(
        *[config[s] for s in ['fftmethod', 'optimizer', 'lr', 'L2', 'dropout', 'batch_norm', 'gradient_clipping',
                              'momentum', 'lr_decay', 'pooling']])
    tb_logger = pl_loggers.TensorBoardLogger(logpath, name=logname)
    print("LOGNAME: {}".format(logname))

    # dataloader initialisation
    dataloader = Dataloader(datapath, batch_size=config['batch_size'], phase_map_file=phase_map_file,
                            fftmethod=config['fftmethod'])
    print("N classes: {}".format(dataloader.num_classes()))

    # model initialisation
    model = Model(num_classes=dataloader.num_classes(), lr=config['lr'], batch_norm=config['batch_norm'],
            dropout = config['dropout'], L2 = config['L2'], momentum=config['momentum'],lr_decay = config['lr_decay'],
                  optimizer=config['optimizer'], poolmethod=config['pooling'])
    trainer = pl.Trainer(logger=tb_logger, max_epochs=config['max_epochs'], tpu_cores=tpu_cores, gpus=gpu_cores,
                         log_every_n_steps=10,
                         callbacks=[EarlyStopping(
                             monitor='val_loss',
                             patience=config['patience'])] if config['early_stopping'] else None)

    # run
    trainer.fit(model, dataloader)
