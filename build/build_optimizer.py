from torch.optim import *

__all__ = ['build_optimizer', 'Adagrad', 'Adadelta', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'LBFGS', 'NAdam', 'RAdam',
           'RMSprop', 'Rprop', 'SGD', 'SparseAdam']


# TODO Solve the problem that **kwargs cannot be passed
def build_optimizer(model, cfg):
    optimizer_name = cfg.optimizer['type']
    optimizer_class = globals()[optimizer_name]
    return optimizer_class(model.parameters(), **cfg.optimizer['setting'])
