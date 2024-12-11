from easydict import EasyDict as edict
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, RandCropByPosNegLabeld

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
# 1) single-modality image training, use csv annotation file
# 2) multi-modality image training, use csv annotation file

__C.general.imseg_list = '/root/projects/Aortic/datasets/test/CTA_AO.csv'

# val set
__C.general.val_list = '/root/projects/Aortic/datasets/test/AO.csv'

# the output of training models and logs
__C.general.save_dir = '/root/projects/Aortic/models/vnet_AO_192/'

# partial preload dir
__C.general.load_dir = ''

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# when finetune from certain model, can choose clear start epoch idx
__C.general.clear_start_epoch = False

# the number of GPUs used in training
__C.general.num_gpus = 2

# random seed used in training (debugging purpose)
__C.general.seed = 0

# wether preload partial params(cascaded for example)
__C.general.partial = False

#####################################
# net
#####################################

__C.net = {}

# the network name
# 1)vnet; 2)vbnet; 3)vbbnet; 4)vbbnet; 5)vnet_lw; 6)vnet_slw;
# 7)se_vbnet; 8)se_vbnet_r8; 9)se_vbnet_r16; 10)se_vbnet_r32;
# 11)se_vbnet_r32_lw; 12)se_vbnet_r32_without_bias
__C.net.name = 'vnet'
# __C.net.name = 'Unet'

# deep supervision config (only vbnet support for deep supervision training now)
__C.net.use_ds = False
# __C.net.use_ds = True

# whether synchronizes bn layer between GPUs
# The batch size of BN layers is __C.train.batchsize if it is True, else __C.train.batchsize // num_gpus which is consistent with DP
# This parameter works only when using distributed training, but it can lead to slower training
__C.net.bn_sync = False

#'fixed_box'  or 'fixed_spacing' or 'new_fixed_box
# used when apply 
__C.net.cropping_method = 'new_fixed_box' 

__C.net.crop_voxel_size = [96, 96, 64] #when cropping_method is fixed_box

##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2

# the resolution on which segmentation is performed
__C.dataset.spacing = [1, 1, 1]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [192, 192, 128]

# data augmentation settings
__C.dataset.aug_config = {
    'aug_prob': 0.5,
    'shift_config': {'shift_prob': 1, 'shift_mm': 10},
    'rot_config': {'rot_prob': 0.5, 'rot_angle_degree': 180, 'rot_axis': [[0, 0, 1]]},
    'flip_config': {'flip_x_prob': 0, 'flip_y_prob': 0, 'flip_z_prob': 0},
    'scale_config': {'scale_prob': 0.5, 'scale_min_ratio': 0.9, 'scale_max_ratio': 1.1, 'scale_isotropic': False},
    'cover_config': {'cover_prob': 0, 'cover_ratio': 0.4, 'mu_value': 20, 'sigma_value': 50},
    'truncate_config': {'trunc_bottom_prob': 0, 'trunc_top_prob': 0, 'trunc_ratio_range': (0.2, 0.5)},
    'brightness_config': {'brightness_prob': 0, 'mul_range': (0.8, 1.2)},
    'contrast_config': {'contrast_prob': 0, 'contrast_range': (0.75, 1.25)},
    'gamma_contrast_config': {'gamma_contrast_prob': 0.0, 'gamma_range': (0.7, 1.5)},
    'gaussian_noise_config': {'gaussian_noise_prob': 0.0, 'noise_scale': (0, 0.05*1024)},
    'gaussian_blur_config': {'gaussian_blur_prob': 0, 'sigma_range': (0.5, 1.5)},
    'simulatelowres_config': {'simulatelowres_prob': 0, 'zoom_range': (0.5, 1)}
}

# the re-sample padding type (0 for zero-padding, 1 for edge-padding)
__C.dataset.pad_t = 0

# the default padding value list for each modality
__C.dataset.default_values = [-1024, 0, 0]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) BOX: sampling crops randomly within segmentation box
# 4) MIXED: sampling crops with mask and box method
__C.dataset.sampling_method = 'MASK'

# the select rate mixed_rate = mask/(box+mask)
__C.dataset.mixed_rate = 0.5

# random select region padding [x_min, y_min, z_min, x_max, y_max, z_max]
__C.dataset.box_sample_padding = [0, 0, 0, 0, 0, 0]

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'NN'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use the minimum and maximum percentiles to normalize image intensities
# 3) NNUnetNormalizer: use mean and stddev calculated from data set to normalize image intensities
#    for example,__C.dataset.crop_normalizers = [AdaptiveNormalizer(min_p=0.01, max_p=0.99, clip=True)]
__C.dataset.crop_normalizers = {"mean":300, "std":400}
####################################
# training loss
####################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity loss, supports binary-class and multi-class segmentation
# BoundaryDice: Boundary loss, supports binary-class and multi-class segmentation
# DoubleLoss: Focal loss(focal_gamma=0 is Cross Entropy loss) and Dice Similarity loss, supports binary-class and multi-class segmentation
# DoubleDiceLoss: Cl_dice Loss and Dice Similarity loss
# clDiceLoss: Cl_dice Loss, binary only
# 

# __C.loss.name = 'clDiceLoss'
# __C.loss.name = 'DoubleDiceLoss'
__C.loss.name = 'DoubleLoss'

# the weight for DoubleLoss/DoubleDiceLoss loss function
# weights will be normalized
__C.loss.double_loss_weight = [0.5, 0.5]

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1, 1]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

# parameters for BoundaryLoss,k=min(epoch_idx / k_slope + 0.01, k_max)
__C.loss.k_slope = 700
__C.loss.k_max = 0.2
__C.loss.level = 20
__C.loss.dim = 3

# parameters for cldice loss
__C.loss.cldice_iter = 15
__C.loss.smooth = 1e-6

# weight for regular term MEEP
__C.loss.reg_w = 0.5

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 701

# the number of samples in a batch
__C.train.batchsize = 4

# the number of threads for IO
__C.train.num_threads = 8

# the learning rate
__C.train.lr = 1e-3

##### 閺傝纭?CosineAnnealing 閸欏倹鏆?T_max,eta_min,last_epoch
##### 閺傝纭?Step            閸欏倹鏆?step_size, gamma, last_epoch
##### 閺傝纭?MultiStep       閸欏倹鏆?milestones, gamma, last_epoch
##### 閺傝纭?Exponential     閸欏倹鏆?gamma, last_epoch
##### last_epoch婵″倹鐏夊▽鈩冩箒鐠佸墽鐤嗛幋鏍偓鍛邦啎缂冾喕璐?1閿涘ast_epoch鐏忓棜顫︾拋鍓х枂娑撶_C.general.resume_epoch
##### 閺傝纭舵潻妯绘箒瀵板牆顦块敍宀冨殰瀹哥湑ytorch閺屻儴顕?# __C.train.lr_scheduler = ReduceLROnPlateau()
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "MultiStepLR"
__C.train.lr_scheduler.params = {'milestones': [120, 200, 350], 'gamma': 0.1, 'last_epoch': -1}

##### 閺傝纭?Adam           閸欏倹鏆?betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
##### 閺傝纭?SGD            閸欏倹鏆?momentum=0, dampening=0, weight_decay=0, nesterov=False
##### 閺傝纭舵潻妯绘箒瀵板牆顦块敍宀冨殰瀹哥湑ytorch閺屻儴顕?
__C.train.optimizer = {}
__C.train.optimizer.name = "AdamW"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False}

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of batches to save model
__C.train.save_epochs = 5
# the number of iterations to evaluation model
__C.train.val_epoch = 5

########################################
# debug parameters
########################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False
