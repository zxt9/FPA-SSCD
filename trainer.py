import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
import torch.nn as nn


class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                val_loader=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch)
    
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.num_classes = self.val_loader.dataset.num_classes
        self.method = self.model.module.method

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])

        self.start_time = time.time()



    def _train_epoch(self, epoch):        
        self.model.train()

        if self.method == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160, position=0)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.method == 'supervised':
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader), (None, None, None, None, None)
            else:
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
                WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
                SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
                target_ul = target_ul.cuda(non_blocking=True)
            A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            self.optimizer.zero_grad()

            total_loss, cur_losses, outputs = self.model(A_l=A_l, B_l=B_l, target_l=target_l, 
                                                         WA_ul=WA_ul, WB_ul=WB_ul, SA_ul=SA_ul, SB_ul=SB_ul)

            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()
                
            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch-1)



            del A_l, B_l, target_l, WA_ul, WB_ul, SA_ul, SB_ul, target_ul
            del total_loss, cur_losses, outputs

            if self.method == 'supervised':
                tbar.set_description('T ({}) | Ls {:.4f} Lu {:.4f} IoU(change-l) {:.3f}| '.format(
                epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1]))
            else:
                tbar.set_description('T ({}) | Ls: {:.4f} Lu: {:.4f} IoU(change-l): {:.3f} IoU(change-ul): {:.3f} F1(ul): {:.3f} Kappa(ul): {:.3f}|'.format(
                epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1], self.class_iou_ul[1], self.f1_ul, self.kappa_ul))

            self.lr_scheduler.step(epoch=epoch-1)



    def _valid_epoch(self, epoch):
        print ('###### EVALUATION ######')

        self.model.eval()
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        tbar = tqdm(self.val_loader, ncols=150)
        with torch.no_grad():
            for batch_idx, (A, B, target,_) in enumerate(tbar):
                target, A, B = target.cuda(non_blocking=True), A.cuda(non_blocking=True), B.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - A.size(2), up_sizes[1] - A.size(3)
                A = F.pad(A, pad=(0, pad_w, 0, pad_h), mode='reflect')
                B = F.pad(B, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(A_l=A, B_l=B)
                output = output[:, :, :H, :W]
                
                # LOSS
                loss = F.cross_entropy(output, target)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union, tp, fp, tn, fn = eval_metrics(output, target, self.num_classes)
                total_inter, total_union = total_inter+inter, total_union+union
                total_tp, total_fp = total_tp+tp, total_fp+fp
                total_tn, total_fn = total_tn+tn, total_fn+fn

                # PRINT INFO
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                P = 1.0 * total_tp / (total_tp + total_fp + np.spacing(1))
                R = 1.0 * total_tp / (total_tp + total_fn + np.spacing(1))
                F1 = 2 * P * R / (P + R + np.spacing(1))
                OA = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + np.spacing(1))
                PRE = (total_tp + total_fn) * (total_tp + total_fp) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2) \
                    + (total_tn + total_fp) * (total_tn + total_fn) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2)
                Kappa = (OA - PRE) / (1 - PRE)

                seg_metrics = {"Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('EVAL ({}) | Loss: {:.3f}, IoU(change): {:.4f}, F1: {:.4f} , Kappa: {:.4f} |'.\
                                        format(epoch, total_loss_val.average, IoU[1], F1, Kappa))

            if (time.time() - self.start_time) / 3600 > 22:
                self._save_checkpoint(epoch, save_best=self.improved)

        return IoU[1]


    def _reset_metrics(self):
        self.loss_l = AverageMeter()
        self.loss_ul  = AverageMeter()
        self.loss_ul_alg  = AverageMeter()
        self.loss_ul_cls  = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.total_tp_l, self.total_fp_l = 0, 0
        self.total_tp_ul, self.total_fp_ul = 0, 0
        self.total_tn_l, self.total_fn_l = 0, 0
        self.total_tn_ul, self.total_fn_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.f1_l, self.f1_ul = 0, 0
        self.kappa_l, self.kappa_ul = 0, 0


    def _update_losses(self, cur_losses):
        if "loss_l" in cur_losses.keys():
            self.loss_l.update(cur_losses['loss_l'].mean().item())
        if "loss_ul" in cur_losses.keys():
            self.loss_ul.update(cur_losses['loss_ul'].mean().item())
        if "loss_ul_alg" in cur_losses.keys():
            self.loss_ul_alg.update(cur_losses['loss_ul_alg'].mean().item())
        if "loss_ul_cls" in cur_losses.keys():
            self.loss_ul_cls.update(cur_losses['loss_ul_cls'].mean().item())


    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['pred_l'], target_l, self.num_classes)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.class_iou_l, self.f1_l, self.kappa_l = seg_metrics_l.values()

        if self.method != 'supervised':
            seg_metrics_ul = eval_metrics(outputs['pred_ul'], target_ul, self.num_classes)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.class_iou_ul, self.f1_ul, self.kappa_ul = seg_metrics_ul.values()
            

    def _update_seg_metrics(self, correct, labeled, inter, union, tp, fp, tn, fn, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
            self.total_tp_l += tp
            self.total_fp_l += fp
            self.total_tn_l += tn
            self.total_fn_l += fn
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union
            self.total_tp_ul += tp
            self.total_fp_ul += fp
            self.total_tn_ul += tn
            self.total_fn_ul += fn


    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
            P = 1.0 * self.total_tp_l / (self.total_tp_l + self.total_fp_l + np.spacing(1))
            R = 1.0 * self.total_tp_l / (self.total_tp_l + self.total_fn_l + np.spacing(1))
            F1 = 2 * P * R / (P + R + np.spacing(1))
            OA  = (self.total_tp_l + self.total_tn_l) / (self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(1))
            PRE = (self.total_tp_l + self.total_fn_l) * (self.total_tp_l + self.total_fp_l) / ((self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(1))**2) \
                + (self.total_tn_l + self.total_fp_l) * (self.total_tn_l + self.total_fn_l) / ((self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(1))**2)
            Kappa = (OA - PRE) / (1 - PRE)

        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
            P = 1.0 * self.total_tp_ul / (self.total_tp_ul + self.total_fp_ul + np.spacing(1))
            R = 1.0 * self.total_tp_ul / (self.total_tp_ul + self.total_fn_ul + np.spacing(1))
            F1 = 2 * P * R / (P + R + np.spacing(1))
            OA = (self.total_tp_ul + self.total_tn_ul) / (self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(1))
            PRE = (self.total_tp_ul + self.total_fn_ul) * (self.total_tp_ul + self.total_fp_ul) / ((self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(1))**2) \
                + (self.total_tn_ul + self.total_fp_ul) * (self.total_tn_ul + self.total_fn_ul) / ((self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(1))**2)
            Kappa = (OA - PRE) / (1 - PRE)

        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "F1": np.round(F1, 4),
            "Kappa": np.round(Kappa, 4)
        }


