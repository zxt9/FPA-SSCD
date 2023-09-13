import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import *
from utils.losses import *
from models.decoder import *
from models.encoder import *


class FPA_ResNet101_CD(BaseModel):
    def __init__(self, num_classes, conf, loss_l=None, loss_alg=None, len_unsper=None, testing=False, pretrained=True):
        self.num_classes = num_classes
        if not testing:
            assert (loss_l is not None) 

        super(FPA_ResNet101_CD, self).__init__()
        self.method = conf['method']

        # Supervised and unsupervised losses        
        self.loss_l         = loss_l
        self.loss_alg       = loss_alg

        if not testing:
            self.unsup_loss_w   = consistency_weight(final_w=1, iters_per_epoch=len_unsper,
                                                     rampup_ends=8)

        # confidence masking (sup mat)
        self.confidence_thr     = conf['confidence_thr']
        print ('thr: ', self.confidence_thr)

        # Create the model
        self.encoder = Encoder_ResNet101(pretrained=pretrained)

        # The main encoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.decoder        = Decoder(upscale, decoder_in_ch, num_classes=num_classes)

    def forward(self, A_l=None, B_l=None, target_l=None, WA_ul=None, WB_ul=None, SA_ul=None, SB_ul=None, 
                target_ul=None, curr_iter=None, epoch=None):
        if not self.training:
            return self.decoder(self.encoder(A_l, B_l))
        input_size  = (A_l.size(2), A_l.size(3))

        # If supervised mode only, return
        if self.method == 'supervised':
            # Supervised loss
            out_l  = self.decoder(self.encoder(A_l, B_l))
            loss_l = self.loss_l(out_l, target_l) 
            curr_losses = {'loss_l': loss_l}
            total_loss = loss_l

            if out_l.shape != A_l.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l}

            return total_loss, curr_losses, outs

        # If semi supervised mode
        else:
            # Supervised loss
            out_l  = self.decoder(self.encoder(A_l, B_l))
            loss_l = self.loss_l(out_l, target_l) 

            # Get main prediction
            weak_out_ul    = self.decoder(self.encoder(WA_ul, WB_ul))
            strong_feat_ul = self.encoder(SA_ul, SB_ul)
            strong_out_ul  = self.decoder(strong_feat_ul)

            # Generate pseudo_label
            weak_prob_ul = F.softmax(weak_out_ul.detach_(), dim=1)
            max_probs, target_ul = torch.max(weak_prob_ul, dim=1)
            mask = max_probs.ge(self.confidence_thr).float()
            loss_ul_cls  = (F.cross_entropy(strong_out_ul, target_ul, reduction='none') * mask).mean()
            loss_ul_alg = self.loss_alg(weak_prob_ul, strong_feat_ul, self.confidence_thr)
            loss_ul = loss_ul_cls + loss_ul_alg
            

            # record loss
            curr_losses = {'loss_l': loss_l}
            curr_losses['loss_ul'] = loss_ul

            if weak_out_ul.shape != WA_ul.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
                weak_out_ul = F.interpolate(weak_out_ul, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l, 'pred_ul': weak_out_ul}

            # Compute the unsupervised loss
            total_loss  = loss_l + loss_ul  
            
            return total_loss, curr_losses, outs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.decoder.parameters())

