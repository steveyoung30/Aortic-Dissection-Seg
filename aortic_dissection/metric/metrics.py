from skimage.morphology import skeletonize, skeletonize_3d, closing, binary_dilation, binary_erosion, square
import GeodisTK
import numpy as np
import surface_distance as sf
import torch
from scipy import ndimage

def binary_FPR(gt, pred, epsilon = 1e-6):
    B = gt.shape[0]
    gt = gt.contiguous().view(B, -1)
    pred = pred.contiguous().view(B, -1)
    intersection = gt * pred
    fp = torch.sum(pred - intersection, dim=1)
    all_negatives = torch.where(gt==0, 1, 0).contiguous().view(B, -1)
    all_negatives = torch.sum(all_negatives, dim=1) 
    fpr = (fp + epsilon) / (all_negatives + epsilon)
    return fpr.mean()

def binary_ASD(mask_gt, mask_pred, spacing = (0.5, 0.5, 0.5)):
    mask_gt = np.where(mask_gt.numpy() > 0 , True, False)
    mask_pred = np.where(mask_pred.numpy() > 0, True, False)
    surface_distances = sf.compute_surface_distances(mask_gt, mask_pred, spacing_mm = spacing)
    avg_surf_dist = sf.compute_average_surface_distance(surface_distances)
    return avg_surf_dist

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    # _, v_p = v_p.max(1)
    v_p = v_p.squeeze().squeeze().cpu().numpy()
    v_l = v_l.squeeze().squeeze().cpu().numpy()
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def binary_dice_score(mask, target, epsilon=1e-6):
    """
    pred: probs from softmax 
    target: binary gt mask
    both should be of shape B, C, D, H, W
    """
    B = mask.shape[0]
    # _, mask = pred.max(dim=1)
    mask = mask.float().contiguous().view(B, -1)
    target = target.float().contiguous().view(B, -1)
    inter = torch.sum(mask * target, dim=1)
    summation = torch.sum(mask, dim=1)+torch.sum(target, dim=1)
    dice = (2*inter + epsilon) / (summation + epsilon)
    return dice.mean().item()

def multiclass_dice_score(pred, target, epsilon=1e-6):
    assert len(pred.shape)==5
    
    B, num_class = pred.shape[0], pred.shape[1]
    all_slices = torch.split(pred, [1]*num_class, dim=1)
    
    total_dice = 0
    for idx in range(num_class):
        pred_i = torch.cat([1 - all_slices[idx], all_slices[idx]], dim=1)
        target_i = (target == idx) * 1
        dice_i = binary_dice_score(pred_i, target_i, epsilon=epsilon)
        total_dice += dice_i
    avg_dice = total_dice / num_class
    return avg_dice

def binary_IoU(pre_mask, gt_mask, epsilon=1e-6):
    """
    Args:
        pre_mask (Tensor[int]): Predicted mask
        gt_mask (Tensor[int]): Ground truth mask

    Returns:
        float: IoU score
    """
    B = pre_mask.shape[0]
    pre_mask = pre_mask.float().contiguous().view(B, -1)
    gt_mask = gt_mask.float().contiguous().view(B, -1)
    inter = torch.sum(pre_mask * gt_mask, dim=1)
    summation = torch.sum(pre_mask, dim=1)+torch.sum(gt_mask, dim=1)
    union = summation - inter
    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean().item()

def binary_dice_iou(mask, target, epsilon=1e-6):
    B = mask.shape[0]
    # _, mask = pred.max(dim=1)
    mask = mask.float().contiguous().view(B, -1)
    target = target.float().contiguous().view(B, -1)
    inter = torch.sum(mask * target, dim=1)
    summation = torch.sum(mask, dim=1)+torch.sum(target, dim=1)
    union = summation - inter
    dice = (2*inter + epsilon) / (summation + epsilon)
    iou = (inter + epsilon) / (union + epsilon)
    return dice.mean(), iou.mean()

def get_edge_points(img, dim=3):
    """
    get edge points of a binary segmentation result
    img ndarray of dimension dim
    """
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_hausdorff95(s, g, spacing=None, image_dim=3):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 5D prob tensor
        g: a 5D binary ground truth mask
        spacing: a list for image spacing, length should be 3 or 2
    """
    _, s = s.max(1)
    s = s.squeeze().squeeze().cpu().numpy()
    g = g.squeeze().squeeze().cpu().numpy()
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    assert image_dim == len(s.shape) == len(g.shape)
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s).astype(np.uint8)

    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    
    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    # print('dist1', dist1)
    # print('dist2', dist2)
    return max(dist1, dist2)