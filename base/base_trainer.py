import torch
from torch.utils import tensorboard
from utils import helpers
import utils.lr_scheduler
import os, json, math, sys, datetime


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:
    def __init__(self, model, resume, config, iters_per_epoch):
        self.model = model
        self.config = config

        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_other_params())},
                            {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                            'lr': config['optimizer']['args']['lr'] / 10}]

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        model_params = sum([i.shape.numel() for i in list(model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer, num_epochs=self.epochs, 
                                        iters_per_epoch=iters_per_epoch)

        # MONITORING
        self.mnt_curr = 0
        self.mnt_best = 0
        self.improved = False

        # CHECKPOINTS & TENSOBOARD
        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        run_name = config['experim_name']
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)
         
        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print ('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print (f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))

        return device, available_gpus


    def train(self):
        # self.mnt_curr = self._valid_epoch(epoch)
        for epoch in range(self.start_epoch, self.epochs+1):
            self._train_epoch(epoch)

            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                self.mnt_curr = self._valid_epoch(epoch)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                self.improved = (self.mnt_curr > self.mnt_best)    
                self.mnt_best = self.mnt_curr if self.improved else self.mnt_best

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = os.path.join(self.checkpoint_dir, 'checkpoint_thr-{}.pth'.format(self.config['model']['confidence_thr']))
        print (f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, 'best_model_thr-{}.pth'.format(self.config['model']['confidence_thr']))
            torch.save(state, filename)
            print ("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        print (f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        try:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print (f'Error when loading: {e}')
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
