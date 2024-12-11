import torch
import torch.nn as nn

from aortic_dissection.loss import *


class DoubleLoss(nn.Module):

    def __init__(self, class_num, double_loss_weight, dice_weights,
                 focal_alpha=None, focal_gamma=2, size_average=True):

        super(DoubleLoss, self).__init__()
        assert isinstance(double_loss_weight, list)
        assert len(double_loss_weight) == 2, "the length of DoubleLoss weight must be 2"

        self.double_loss_weight = double_loss_weight
        self.focal_loss_func = FocalLoss(class_num=class_num, alpha=focal_alpha, gamma=focal_gamma)
        if class_num >= 2:
            self.dice_loss_func = MultiDiceLoss(num_class=class_num, weights=dice_weights)
        else:
            raise ValueError('invalid class num')

    def forward(self, inputs, targets):
        weight = torch.FloatTensor(self.double_loss_weight)
        weight = weight.unsqueeze(1)
        weight = weight / weight.sum()
        weight = weight.cuda()

        focal_loss = self.focal_loss_func(inputs, targets)
        # if isinstance(self.dice_loss_func, BinaryDiceLoss):
        #     dice_loss = self.dice_loss_func.apply(inputs, targets)
        # else:
        dice_loss, _ = self.dice_loss_func(inputs, targets)
        dice_loss = dice_loss.cuda()
        focal_loss = focal_loss.cuda()
        print('dcl',dice_loss)
        print('fl',focal_loss)

        return focal_loss * weight[0] + dice_loss * weight[1]


class DoubleDiceLoss(nn.Module):

    def __init__(self, class_num, double_loss_weight, dice_weights, iter=3, smooth=1e-8):

        super(DoubleDiceLoss, self).__init__()
        assert isinstance(double_loss_weight, list)
        assert len(double_loss_weight) == 2, "the length of DoubleLoss weight must be 2"

        self.double_loss_weight = double_loss_weight
        self.cl_dice_loss_func = soft_cldice_loss(iter, smooth)
        if class_num >= 2 :
            self.dice_loss_func = MultiDiceLoss(num_class=class_num, weights=dice_weights) 
        else:
            raise ValueError('invalid class num')

    def forward(self, inputs, targets):
        weight = torch.FloatTensor(self.double_loss_weight)
        weight = weight.unsqueeze(1)
        weight = weight / weight.sum()
        weight = weight.cuda()

        cl_dice_loss = self.cl_dice_loss_func(inputs, targets)
        # if isinstance(self.dice_loss_func, BinaryDiceLoss):
        #     dice_loss = self.dice_loss_func.apply(inputs, targets)
        # else:
        dice_loss, _ = self.dice_loss_func(inputs, targets)
        dice_loss = dice_loss.cuda()
        cl_dice_loss = cl_dice_loss.cuda()
        print('dc_loss',dice_loss)
        print('cl_dc_loss',cl_dice_loss)

        return cl_dice_loss * weight[0] + dice_loss * weight[1]

        
def create_loss_local(cfg):
    """ create loss function according to configuration object """

    assert "obj_weight" in cfg.loss.keys(), "please set __C.loss.obj_weight in train config file"
    assert len(cfg.loss.obj_weight) == cfg.dataset.num_classes

    if cfg.loss.name == 'FocalLoss':
        loss_func = FocalLoss(
            class_num=cfg.dataset.num_classes,
            alpha=cfg.loss.obj_weight,
            gamma=cfg.loss.focal_gamma)

    elif cfg.loss.name == 'DiceLoss':
        dice_loss_func = MultiDiceLoss(
            num_class=cfg.dataset.num_classes,
            weights=cfg.loss.obj_weight)
        loss_func = lambda inputs, targets: dice_loss_func(inputs, targets)[0]

    elif cfg.loss.name == 'BoundaryLoss':
        loss_func = BoundarySoftDice(
            k=cfg.loss.k_max,
            weights=cfg.loss.obj_weight,
            num_class=cfg.dataset.num_classes,
            level=cfg.loss.level, dim=cfg.loss.dim)

    elif cfg.loss.name == 'DoubleLoss':
        assert "double_loss_weight" in cfg.loss.keys(),\
            "please set __C.loss.double_loss_weight in train config file"
        loss_func = DoubleLoss(
            class_num=cfg.dataset.num_classes,
            double_loss_weight=cfg.loss.double_loss_weight,
            dice_weights=cfg.loss.obj_weight,
            focal_alpha=cfg.loss.obj_weight,
            focal_gamma=cfg.loss.focal_gamma)
    elif cfg.loss.name == 'DoubleDiceLoss':
        assert "double_loss_weight" in cfg.loss.keys(),\
            "please set __C.loss.double_loss_weight in train config file"
        loss_func = DoubleDiceLoss(
            class_num=cfg.dataset.num_classes,
            double_loss_weight=cfg.loss.double_loss_weight,
            dice_weights=cfg.loss.obj_weight,
            iter=cfg.loss.cldice_iter)
    elif cfg.loss.name == 'clDiceLoss':
        loss_func = soft_cldice_loss(iter_=cfg.loss.num_iters, smooth=cfg.loss.smooth)
    else:
        raise ValueError('Unknown loss function')

    return loss_func