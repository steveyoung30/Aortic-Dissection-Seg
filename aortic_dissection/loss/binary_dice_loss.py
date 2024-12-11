import torch
from torch.autograd import Function
import numpy as np

class BinaryDiceLoss(Function):
    """ Dice Loss for binary segmentation
        Dice loss = 1 - Dice (minimize loss, maximize dice)

        The computation of Dice is slightly different in the forward and backward passes

        A is prediction result, B is ground-truth (binary label).

        In the FORWARD pass:
        The Dice is computed exactly as its definition. Binarized A first.
        Intersect = dot(A, B)
        Sum = sum(A) + sum(B)
        Dice = 2 * Intersect / Sum

        In the BACKWARD pass:
        The gradient is derived from the the following definition. A is softmax result, B is binary label.
        Dice = 2 * Intersect / Sum
        Intersect = dot(A, B)
        Sum = dot(A, A) + dot(B, B)    (interesting definition)

        Partially expand the derivative:
        d(Dice)/dA = 2 * [d(Intersect)/dA * Sum - Intersect * d(Sum)/dA] / (Sum)^2
        d(Intersect)/dA = B
        d(Sum)/dA = 2A

        Combine the above three definitons together
        d(Dice)/dA = 2 * [B * Sum - 2A * Intersect] / (Sum)^2

        The values Intersect and Sum are used from the forward pass.
    """

    @staticmethod
    def forward(ctx, input, target, save=True, epsilon=1e-6):

        batchsize = input.size(0)

        # convert probability to binary label using maximum probability
        _, input_label = input.max(1)

        # convert to floats
        input_label = input_label.float()
        target_label = target.float()

        if save:
            # save float version of target for backward
            ctx.save_for_backward(input, target_label)

        # convert to 1D
        # input_label = input_label[target_label.squeeze(1) != -1]
        # target_label = target_label[target_label != -1]
        
        # diff = (target_label != -1).sum() % batchsize
        # if diff != 0:
        #     input_label = torch.cat((input_label, torch.zeros(batchsize - diff).cuda()))
        #     target_label = torch.cat((target_label, torch.zeros(batchsize - diff).cuda()))
        
        input_label = input_label.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        target_label_patch = target_label.view(1, -1)

        target_label_sum = target_label.sum(dim=1)
        pos_case_num = (target_label_sum > 0).sum()
        if pos_case_num == 0:
            coe = torch.FloatTensor(np.ones(batchsize)).cuda()
        else:
            target_label_sum = target_label.sum(dim=1)
            voxel_sum = target_label_patch.sum()
            coe = voxel_sum / target_label_sum / pos_case_num
            coe[target_label_sum==0] = 1
        
        # compute dice score
        ctx.intersect = torch.sum((input_label *coe.unsqueeze(1) * target_label).view(1, -1), dim=1)
        input_area = torch.sum((input_label * coe.unsqueeze(1)).view(1, -1), dim=1)
        target_area = torch.sum((target_label * coe.unsqueeze(1)).view(1, -1), dim=1)

        ctx.sum = input_area + target_area + 2 * epsilon

        # intersect = torch.sum(ctx.intersect)
        # vsum = torch.sum(ctx.sum)
        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = 1 - 2 * ctx.intersect / ctx.sum
        
        # batch_loss = 1 - 2 * intersect / vsum
        if pos_case_num == 0:
            batch_loss = torch.zeros_like(batch_loss).cuda()
        # if batch_loss < 0:
        #     batch_loss = epsilon
        # batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        return loss

    @staticmethod
    def backward(ctx, grad_out):
        # gradient computation
        # d(Dice) / dA = [2 * target * Sum - 4 * input * Intersect] / (Sum) ^ 2
        #              = (2 * target / Sum) - (4 * input * Intersect / Sum^2)
        #              = (2 / Sum) * target - (4 * Intersect / Sum^2) * Input
        #              = a * target - b * input
        #
        # DiceLoss = 1 - Dice
        # d(DiceLoss) / dA = -a * target + b * input
        input, target = ctx.saved_tensors
        intersect, sum = ctx.intersect, ctx.sum

        a = 2 / sum
        b = 4 * intersect / sum / sum

        batchsize = a.size(0)
        a = a.view(batchsize, 1, 1, 1, 1)
        b = b.view(batchsize, 1, 1, 1, 1)

        grad_diceloss = -a * target + b * input[:, 1:2]

        # TODO for target=0 (background), if whether b is close 0, add epsilon to b

        # sum gradient of foreground and background probabilities should be zero
        
        grad_input = torch.cat((grad_diceloss * -grad_out.item(),
                            grad_diceloss *  grad_out.item()), 1)

        # 1) gradient w.r.t. input, 2) gradient w.r.t. target
        return grad_input, None

def cal_dsc_loss(input, target):

    batchsize = input.size(0)

    _, input_label = input.data.max(1)
    input_label = input_label.float().view(batchsize, -1)
    target_label = target.data.view(batchsize, -1)

    intersect = torch.sum(input_label * target_label, 1)

    epsilon = 1e-6
    input_area = torch.sum(input_label, 1)
    target_area = torch.sum(target_label, 1)
    sum = input_area + target_area + 2 * epsilon

    batch_loss = 1.0 - 2.0 * intersect / sum
    batch_loss[target_area == 0] = 0

    return batch_loss

# import torch
# from torch.autograd import Function
# from md.mdpytorch.utils.pytorch_version import minor_version

# class BinaryDiceLoss(Function):
#     """ Dice Loss for binary segmentation
#         Dice loss = 1 - Dice (minimize loss, maximize dice)

#         The computation of Dice is slightly different in the forward and backward passes

#         A is prediction result, B is ground-truth (binary label).

#         In the FORWARD pass:
#         The Dice is computed exactly as its definition. Binarized A first.
#         Intersect = dot(A, B)
#         Sum = sum(A) + sum(B)
#         Dice = 2 * Intersect / Sum

#         In the BACKWARD pass:
#         The gradient is derived from the the following definition. A is softmax result, B is binary label.
#         Dice = 2 * Intersect / Sum
#         Intersect = dot(A, B)
#         Sum = dot(A, A) + dot(B, B)    (interesting definition)

#         Partially expand the derivative:
#         d(Dice)/dA = 2 * [d(Intersect)/dA * Sum - Intersect * d(Sum)/dA] / (Sum)^2
#         d(Intersect)/dA = B
#         d(Sum)/dA = 2A

#         Combine the above three definitons together
#         d(Dice)/dA = 2 * [B * Sum - 2A * Intersect] / (Sum)^2

#         The values Intersect and Sum are used from the forward pass.
#     """

#     @staticmethod
#     def forward(ctx, input, target, save=True, epsilon=1e-6):

#         batchsize = input.size(0)
#         ctx.batchsize = batchsize

#         # convert probability to binary label using maximum probability
#         _, input_label = input.max(1)

#         # convert to floats
#         input_label = input_label.float()
#         target_label = target.float()

#         if save:
#             # save float version of target for backward
#             ctx.save_for_backward(input, target_label)

#         # convert to 1D
#         input_label = input_label.view(batchsize, -1)
#         target_label = target_label.view(batchsize, -1)
        
#         # input_label = input_label.view(1, -1)
#         # target_label = target_label.view(1, -1)
#         # input_label = input_label[target_label != -1]
#         # target_label = target_label[target_label != -1]
        
#         # compute dice score
#         # ctx.intersect = torch.sum(input_label * target_label, 1)
#         # input_area = torch.sum(input_label, 1)
#         # target_area = torch.sum(target_label, 1)
        
#         ctx.intersect = torch.sum(input_label * target_label)
#         input_area = torch.sum(input_label)
#         target_area = torch.sum(target_label)

#         ctx.sum = input_area + target_area + 2 * epsilon

#         # batch dice loss and ignore dice loss where target area = 0
#         batch_loss = 1 - 2 * ctx.intersect / ctx.sum
#         batch_loss[target_area == 0] = 0
#         loss = batch_loss.mean()

#         if minor_version() < 4:
#             return torch.FloatTensor(1).fill_(loss)
#         else:
#             return loss

#     @staticmethod
#     def backward(ctx, grad_out):

#         # gradient computation
#         # d(Dice) / dA = [2 * target * Sum - 4 * input * Intersect] / (Sum) ^ 2
#         #              = (2 * target / Sum) - (4 * input * Intersect / Sum^2)
#         #              = (2 / Sum) * target - (4 * Intersect / Sum^2) * Input
#         #              = a * target - b * input
#         #
#         # DiceLoss = 1 - Dice
#         # d(DiceLoss) / dA = -a * target + b * input
#         input, target = ctx.saved_tensors
#         intersect, sum = ctx.intersect, ctx.sum

#         a = 2 / sum
#         b = 4 * intersect / sum / sum

#         # batchsize = a.size(0)
#         batchsize = ctx.batchsize
#         a = a.view(batchsize, 1, 1, 1, 1)
#         b = b.view(batchsize, 1, 1, 1, 1)

#         grad_diceloss = -a * target + b * input[:, 1:2]

#         # TODO for target=0 (background), if whether b is close 0, add epsilon to b

#         # sum gradient of foreground and background probabilities should be zero
#         if minor_version() < 4:
#             grad_input = torch.cat((grad_diceloss * -grad_out[0],
#                                     grad_diceloss * grad_out[0]), 1)
#         else:
#             grad_input = torch.cat((grad_diceloss * -grad_out.item(),
#                                 grad_diceloss *  grad_out.item()), 1)

#         # 1) gradient w.r.t. input, 2) gradient w.r.t. target
#         return grad_input, None

# def cal_dsc_loss(input, target):

#     batchsize = input.size(0)

#     _, input_label = input.data.max(1)
#     input_label = input_label.float().view(batchsize, -1)
#     target_label = target.data.view(batchsize, -1)

#     intersect = torch.sum(input_label * target_label, 1)

#     epsilon = 1e-6
#     input_area = torch.sum(input_label, 1)
#     target_area = torch.sum(target_label, 1)
#     sum = input_area + target_area + 2 * epsilon

#     batch_loss = 1.0 - 2.0 * intersect / sum
#     batch_loss[target_area == 0] = 0

#     return batch_loss
