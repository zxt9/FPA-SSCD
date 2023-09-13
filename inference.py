import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.metrics import eval_metrics, AverageMeter

from matplotlib import pyplot as plt
from utils.helpers import DeNormalize
import time
import PIL


def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    H, W        = (image_A.size(2), image_A.size(3))
    upsize      = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample    = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w= upsize[0] - H, upsize[1] - W
    image_A     = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B     = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        scaled_prediction = F.softmax(scaled_prediction, dim=1)

        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions  = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction   = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [1.0,1.25]

    # DATA LOADER
    config['val_loader']["batch_size"]  = 1
    config['val_loader']["num_workers"] = 1
    config['val_loader']["split"]       = "test"
    config['val_loader']["shuffle"]     = False
    config['val_loader']['data_dir']    = args.Dataset_Path
    config['val_loader']["aug_type"]  = 'all'
    loader = dataloaders.CDDataset(config['val_loader'])
    num_classes = 2
    palette     = get_voc_pallete(num_classes)

    # MODEL
    dataset = args.Dataset_Path.split('/')[-1]
    percent = config['percent']
    method = config['model']['method'] 
    backbone = config['model']['backbone'] 
    if backbone == 'ResNet50':
        model = models.FPA_ResNet50_CD(num_classes=num_classes, conf=config['model'], testing=True)
    elif backbone == 'HRNet':
        model = models.FPA_HRNet_CD(num_classes=num_classes, conf=config['model'], testing=True)
    elif backbone == 'ResNet101':
        model = models.FPA_ResNet101_CD(num_classes=num_classes, conf=config['model'], testing=True)
    print(f'\n{model}\n')
    checkpoint = torch.load(args.model)

    try:
        print("Loading the state dictionery of {} ...".format(args,model))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    print (config)


    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=120)
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    for index, data in enumerate(tbar):
        image_A, image_B, label, [image_A_path, image_B_path, label_path] = data
        image_A = image_A.cuda()
        image_B = image_B.cuda()
        label   = label.cuda()
        image_A_path = image_A_path[0]
        image_B_path = image_B_path[0]
        label_path = label_path[0]
        image_id = image_A_path.split('/')[-1].split('\\')[-1].strip('.jpg').strip('.png')

        #PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image_A, image_B, scales, num_classes)

        heatmap_im = np.squeeze(output[1:,::])
        heatmap_im = cv2.applyColorMap(np.uint8(255*heatmap_im), cv2.COLORMAP_JET)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        #Calculate metrics
        output = torch.from_numpy(output).cuda()
        label[label>=1] = 1
        output = torch.unsqueeze(output, 0)
        label  = torch.unsqueeze(label, 0)
        correct, labeled, inter, union, tp, fp, tn, fn  = eval_metrics(output, label, num_classes)

        total_inter, total_union        = total_inter+inter, total_union+union
        total_correct, total_label      = total_correct+correct, total_label+labeled
        total_tp, total_fp = total_tp+tp, total_fp+fp
        total_tn, total_fn = total_tn+tn, total_fn+fn

        
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        P = 1.0 * total_tp / (total_tp + total_fp + np.spacing(1))
        R = 1.0 * total_tp / (total_tp + total_fn + np.spacing(1))
        F1 = 2 * P * R / (P + R + np.spacing(1))
        OA = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + np.spacing(1))
        PRE = (total_tp + total_fn) * (total_tp + total_fp) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2) \
            + (total_tn + total_fp) * (total_tn + total_fn) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2)
        Kappa = (OA - PRE) / (1 - PRE)
        TPR = total_tp / (total_tp + total_fn + np.spacing(1))
        TNR = total_tn / (total_tn + total_fp + np.spacing(1))
        tbar.set_description('Test Results | IoU(change): {:.4f}, F1: {:.4f} , Kappa: {:.4f}, TPR: {:.4f}, TNR: {:.4f} |'.\
                                        format( IoU[1], F1, Kappa, TPR, TNR))


    
    #Printing average metrics on test-data
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)

    P = 1.0 * total_tp / (total_tp + total_fp + np.spacing(1))
    R = 1.0 * total_tp / (total_tp + total_fn + np.spacing(1))
    F1 = 2 * P * R / (P + R + np.spacing(1))
    OA = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + np.spacing(1))
    PRE = (total_tp + total_fn) * (total_tp + total_fp) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2) \
        + (total_tn + total_fp) * (total_tn + total_fn) / ((total_tp + total_fp + total_tn + total_fn + np.spacing(1))**2)
    Kappa = (OA - PRE) / (1 - PRE)

    TPR = total_tp / (total_tp + total_fn + np.spacing(1))
    TNR = total_tn / (total_tn + total_fp + np.spacing(1))

    mIoU = IoU.mean()


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='./saved/WHU-CD/ResNet50/FPA_5/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default='./saved/WHU-CD/ResNet50/FPA_5/best_model_thr-0.95.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument( '--save', action='store_true', help='Save images')
    parser.add_argument('--Dataset_Path', default="../data/WHU-CD-256", type=str,
                        help='Path to dataset WHU-CD')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

