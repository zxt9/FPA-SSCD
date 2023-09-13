import os
import json
import argparse
import torch
import dataloaders
import models
import math
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import  CE_loss, Alg_loss

def main(config, resume, gpu, aug_type='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.manual_seed(42)

    config['val_loader']['aug_type'] = aug_type
    method = config['model']['method'] + '_' + aug_type
    config['train_supervised']['aug_type'] = aug_type
    config['train_unsupervised']['aug_type'] = aug_type

    backbone = config['model']['backbone'] 
    config['experim_name'] = config['experim_name'].replace('method', method)
    config['experim_name'] = config['experim_name'].replace('percent', str(config['percent'] ))
    config['trainer']['save_dir'] = config['trainer']['save_dir'].replace('backbone', backbone)
    print (config)

    # DATA LOADERS
    config['train_supervised']['percnt_lbl'] = config["percent"]
    config['train_unsupervised']['percnt_lbl'] = config["percent"]
    supervised_loader = dataloaders.CDDataset(config['train_supervised'])
    unsupervised_loader = dataloaders.CDDataset(config['train_unsupervised'])
    print ('supervised: ', len(supervised_loader))
    print ('unsupervised: ', len(unsupervised_loader))
    val_loader = dataloaders.CDDataset(config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    loss_l = CE_loss
    loss_alg = Alg_loss

    # MODEL
    if backbone == 'ResNet50':
        model = models.FPA_ResNet50_CD(num_classes=val_loader.dataset.num_classes, 
                                        conf=config['model'],
                                        loss_l=loss_l,
                                        loss_alg=loss_alg,
                                        len_unsper=len(unsupervised_loader))
    elif backbone == 'HRNet':
        model = models.FPA_HRNet_CD(num_classes=val_loader.dataset.num_classes, 
                                     conf=config['model'],
                                     loss_l=loss_l,
                                     loss_alg=loss_alg,
                                     len_unsper=len(unsupervised_loader))
    elif backbone == 'ResNet101':
        model = models.FPA_ResNet101_CD(num_classes=val_loader.dataset.num_classes, 
                                         conf=config['model'],
                                         loss_l=loss_l,
                                         loss_alg=loss_alg,
                                         len_unsper=len(unsupervised_loader))
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_WHU.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-a', '--aug_type', default='all', type=str,
                        help='augmentation type of FPA')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-g', '--gpu', default=7, type=int,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    config = json.load(open(args.config))
    gpu = str(args.gpu)
    aug_type = args.aug_type
    main(config, args.resume, gpu, aug_type)
