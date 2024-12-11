import torch
import torch.nn as nn
import numpy as np
from aortic_dissection.loss.binary_dice_loss import BinaryDiceLoss

class MultiDiceLoss(nn.Module):
    """
    Dice Loss for egmentation(include binary segmentation and multi label segmentation)
    This class is generalization of BinaryDiceLoss
    """
    def __init__(self, weights, num_class):
        """
        :param weights: weight for each class dice loss
        :param num_class: the number of class
        """
        super(MultiDiceLoss, self).__init__()
        self.num_class = num_class

        assert len(weights) == self.num_class, "the length of weight must equal to num_class"
        self.weights = torch.FloatTensor(weights)
        self.weights = self.weights/self.weights.sum()
        self.weights = self.weights.cuda()

    def forward(self, input_tensor, target):
        """
        :param input_tensor: network output tensor
        :param target: ground truth
        :return: weighted dice loss and a list for all class dice loss, expect background
        """
        dice_losses = []
        weight_dice_loss = 0
        all_slice = torch.split(input_tensor, [1] * self.num_class, dim=1)

        for i in range(self.num_class):
            # prepare for calculate label i dice loss
            
            # remove negtive voxels
            
            slice_i = torch.cat([1 - all_slice[i], all_slice[i]], dim=1)
            if i == 0:
                target_i = (target == i) * 1
            else:
                target_i = (target == i) * 1 + (target == -i) * -1

            # BinaryDiceLoss save forward information for backward
            # so we can't use one BinaryDiceLoss for all classes
            dice_function = BinaryDiceLoss()
            dice_i_loss = dice_function.apply(slice_i, target_i)

            # save all classes dice loss and calculate weighted dice
            dice_losses.append(dice_i_loss)
            weight_dice_loss += dice_i_loss * self.weights[i]

        return weight_dice_loss, [dice_loss.item() for dice_loss in dice_losses]
