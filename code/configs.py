'''
file to hold the configs to run VGGnet (and possibly other nets).
'''


config = {
            # learning params
            'lr': 1e-3,
            'batch_size': 32,
            'early_stopping': True,
            'optimizer': 'SGD',
            'max_epochs': 100,
            'early_stopping': True,
            'patience': 3, # tolerance for early stopping

            #, regularization
            'dropout': 0.0,
            'L2': 0.0,
            'batch_norm': True,
            'gradient_clipping': 0.0,

}
