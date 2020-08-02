from __future__ import absolute_import
from __future__ import division
import cv2
import torch
import torch.nn as nn
from keras.models import load_model
# from MAMC import mamc_loss
import numpy as np
from OSME import osme
from keras.optimizers import SGD


def torchmamc(inputs, targets):
    """
    Args:
        inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, feat_dim).
        targets (torch.LongTensor): ground truth labels with shape (num_classes).
    """

    # b, p, _ = inputs.size()
    '''
    b = 8
    p = 2
    n = b*p

    inputs = inputs.reshape((b, p, 1024))
    '''
    b = 4
    p = 2
    n = b * p
    
    inputs = inputs.reshape((b, p, 2))

    #
    inputs = inputs.contiguous().view(n, -1)  # 将输入的向量变为一维


    targets = torch.repeat_interleave(targets, p)
    parts = torch.arange(p).repeat(b)
    prod = torch.mm(inputs, inputs.t())
    # if self.use_gpu:parts = parts.cuda()

    same_class_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    same_atten_mask = parts.expand(n, n).eq(parts.expand(n, n).t())

    s_sasc = same_class_mask & same_atten_mask
    s_sadc = (~same_class_mask) & same_atten_mask
    s_dasc = same_class_mask & (~same_atten_mask)
    s_dadc = (~same_class_mask) & (~same_atten_mask)

    # For each anchor, compute equation (11) of paper
    loss_sasc = 0
    loss_sadc = 0
    loss_dasc = 0
    for i in range(n):
        # loss_sasc
        pos = prod[i][s_sasc[i]]
        neg = prod[i][s_sadc[i] | s_dasc[i] | s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)

        print(torch.exp(neg - pos))
        loss_sasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

        # loss_sadc
        pos = prod[i][s_sadc[i]]
        neg = prod[i][s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)

        loss_sadc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))


        # loss_dasc
        pos = prod[i][s_dasc[i]]
        neg = prod[i][s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)

        loss_dasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

    return (loss_sasc + loss_sadc + loss_dasc) / n




