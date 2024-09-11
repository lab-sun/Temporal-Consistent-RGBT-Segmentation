import torch
import torch.nn as nn

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
   
    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


import torch
import numpy as np
import random
import os
from config import config

def prepare_environment(args):
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    print("\nPyTorch Version:", torch.__version__)
    print("GPU Count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device(), '\n')


# from model.segformer.builder import EncoderDecoder as segmodel

# def init_model(config, args):
#     model = segmodel(cfg=config, encoder_name=config.backbone, decoder_name='MLPDecoder', norm_layer=torch.nn.BatchNorm2d)
#     if args.gpu >= 0: 
#         model.cuda(args.gpu)
#     return model


from util.lr_policy import WarmUpPolyLR

def init_optimizer(model, config):
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, lr=config.lr)
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    print(config.lr)
    return optimizer, lr_policy



from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
def setup_data_loaders(config):
    train_dataset = MF_dataset(data_dir=config.dataset_path, split='train')
    val_dataset = MF_dataset(data_dir=config.dataset_path, split='val')
    test_dataset = MF_dataset(data_dir=config.dataset_path, split='test')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


import os, stat, shutil
from torch.utils.tensorboard import SummaryWriter

def prepare_directories(args):
    # preparing folders
    if os.path.exists("./CMX_mit_b2"):
        shutil.rmtree("./CMX_mit_b2")
    weight_dir = os.path.join("./CMX_mit_b2", args.model_name)
    os.makedirs(weight_dir)

    os.chmod(weight_dir, stat.S_IRWXO)  
    writer = SummaryWriter("./CMX_mit_b2/tensorboard_log")
    os.chmod("./CMX_mit_b2/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./CMX_mit_b2", stat.S_IRWXO)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    return weight_dir, writer