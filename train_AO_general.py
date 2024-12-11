from __future__ import print_function
from builtins import input
import argparse
import importlib
import os
import sys
import time
import math
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import yaml

import torch
import torch.nn as nn
from torch.backends import cudnn

from torch.utils.data import DataLoader

# from md.image3d.python.frame3d import Frame3d
# from md.image3d.python.image3d_io import write_image
# from md.mdpytorch.utils.tensor_tools import ToImage, ToTensor
# from md.utils.python.logging_helpers import setup_logger
# from md.utils.python.file_tools import load_module_from_disk
# from md_segmentation3d.utils.vseg_loss import create_loss_functon

import monai
from monai.transforms import Compose, EnsureChannelFirstd, RandCropByPosNegLabeld, Spacingd
from monai.transforms import ToTensord, NormalizeIntensity, ThresholdIntensityd, ScaleIntensityRanged, ScaleIntensityRanged
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, Dataset

from aortic_dissection.metric import *
from aortic_dissection.dataset.dataset import SegmentationDataset, Val_dataset
from aortic_dissection.utils.loss_utils import create_loss_local
from aortic_dissection.utils.train_utils import EpochConcateSampler
from aortic_dissection.utils.plot_loss import plot_loss

import easydict as edict

class Trainer:
    def __init__():
        pass
    def validate():
        pass
    def train():
        pass

def create_loss(cfg):
    try:
        loss = create_loss_local(cfg)
    except Exception as e:
        loss = importlib.import_module("monai.losses." + cfg.loss.name)
    return loss


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return edict(config)


def create_net(cfg):
    net = importlib.import_module("aortic_dissection.network." + cfg.net.name)
    return net


def setup_logger(log_file, logger_name='Train Logger'):

    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别为DEBUG，允许所有消息通过logger

    # 创建一个handler，用于写入日志文件，只记录ERROR及以上级别的日志
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.ERROR)

    # 再创建一个handler，用于输出到控制台，从INFO级别开始记录
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_transform(dataset_cfg):
    spacing, crop_size = dataset_cfg.spacing, dataset_cfg.crop_size
    transform = Compose([
    EnsureChannelFirstd(keys=["image", "mask"]),  # 确保通道维度位于最前面
    Spacingd(
        keys=["image", "label"],  # 指定需要重采样的键
        pixdim=spacing,  # 目标体素间距
        mode=("trilinear", "nearest"),  # 图像和标签的不同插值模式
        align_corners=(True, True)  # 对齐角点参数
    ),
    NormalizeIntensity(subtrahend=300, divisor=400), # Fixed Mean Std normalization
    ThresholdIntensityd(keys=['image'], threshold=-1., above=True, cval=-1.), # intensity < threshold（对应above=True）的置为cval
    ThresholdIntensityd(keys=['image'], threshold=1., above=False, cval=1.),   # intensity > threshold(对应above=False)的值置为cval
    # lambda x: (x - crop_normalizers["mean"]) / crop_normalizers["std"], # fixed mean std 归一化
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",  # 指定哪个键作为标签用于指导裁剪
        spatial_size=crop_size,  # 裁剪的尺寸
        pos=3, neg=1,  # 正负样本的比例
        num_samples=1,  # 每次迭代生成多少个样本
        # image_key="image"  # 如果需要考虑图像的边界限制，请指定此参数
    ),
    ToTensord(keys=["image", "mask"])  # 转换为PyTorch张量
])
    return transform


def worker_init(worker_idx):
    """
    The worker initialization function takes the worker id (an int in "[0,
    num_workers - 1]") as input and does random seed initialization for each
    worker.
    :param worker_idx: The worker index.
    :return: None.
    """
    MAX_INT = sys.maxsize
    worker_seed = np.random.randint(int(np.sqrt(MAX_INT))) + worker_idx
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

def save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality, max_vscore, last_val_epoch):
    """ save model and parameters into a checkpoint file (.pth)

    :param net: the network object
    :param opt: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :param num_modality: the number of image modalities
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)

    filename = os.path.join(chk_folder, 'params.pth')
    opt_filename = os.path.join(chk_folder, 'optimizer.pth')

    state = {'epoch':             epoch_idx,
             'batch':             batch_idx,
             'net':               cfg.net.name,
             'use_ds':            cfg.net.get('use_ds', False),
             'max_stride':        max_stride,
             'state_dict':        net.state_dict(),
             'spacing':           cfg.dataset.spacing,
             'interpolation':     cfg.dataset.interpolation,
             'pad_t':             cfg.dataset.pad_t,
             'default_values':    cfg.dataset.default_values,
             'in_channels':       num_modality,
             'out_channels':      cfg.dataset.num_classes,
             'crop_normalizers':  [normalizer.to_dict() for normalizer in cfg.dataset.crop_normalizers],
             'crop_size':         cfg.dataset.crop_size,
             'last_val_epoch':    last_val_epoch,
             'best_vscore':       max_vscore
             }

    # save python check point
    torch.save(state, filename)

    # save python optimizer state
    torch.save(opt.state_dict(), opt_filename)

    # save template parameter ini file
    # ini_file = os.path.join(os.path.dirname(__file__), 'config', 'params.ini')
    # shutil.copy(ini_file, os.path.join(cfg.general.save_dir, 'params.ini'))

    # copy config file
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))

def load_checkpoint(epoch_idx, net, opt, save_dir):
    """ load network parameters from directory

    :param epoch_idx: the epoch idx of model to load
    :param net: the network object
    :param opt: the optimizer object
    :param save_dir: the save directory
    :return: loaded epoch index, loaded batch index
    """
    # load network parameters
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    assert os.path.isfile(chk_file), 'checkpoint file not found: {}'.format(chk_file)

    state = torch.load(chk_file)
    net.load_state_dict(state['state_dict'])

    # load optimizer state
    opt_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'optimizer.pth')
    assert os.path.isfile(opt_file), 'optimizer file not found: {}'.format(chk_file)

    opt_state = torch.load(opt_file)
    opt.load_state_dict(opt_state)

    return state['epoch'], state['batch'], state['last_val_epoch'], state['best_vscore']


def load_checkpoint_partial(epoch_idx, net, save_dir): 
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    assert os.path.isfile(chk_file), 'checkpoint file not found: {}'.format(chk_file)

    state = torch.load(chk_file)
    net.load_state_dict(state['state_dict'], strict=False)

    # load optimizer state
    # opt_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'optimizer.pth')
    # assert os.path.isfile(opt_file), 'optimizer file not found: {}'.format(chk_file)

    # opt_state = load_pytorch_model(opt_file)
    # opt.load_state_dict(opt_state, strict=False)

    return state['epoch'], state['batch']

# def save_intermediate_results(idxs, crops, masks, outputs, frames, file_names, out_folder):
#     """ save intermediate results to training folder

#     :param idxs: the indices of crops within batch to save
#     :param crops: the batch tensor of image crops
#     :param masks: the batch tensor of segmentation crops
#     :param outputs: the batch tensor of output label maps
#     :param frames: the batch frames
#     :param file_names: the batch file names
#     :param out_folder: the batch output folder
#     :return: None
#     """
#     if not os.path.isdir(out_folder):
#         os.makedirs(out_folder)

#     for i in idxs:

#         case_out_folder = os.path.join(out_folder, file_names[i])
#         if not os.path.isdir(case_out_folder):
#             os.makedirs(case_out_folder)

#         frame = Frame3d()
#         frame.from_numpy(frames[i].numpy())

#         if crops is not None:
#             images = ToImage()(crops[i])
#             for modality_idx, image in enumerate(images):
#                 image.set_frame(frame)
#                 write_image(image, os.path.join(case_out_folder, 'crop_{}.nii.gz'.format(modality_idx)))

#         if masks is not None:
#             mask = ToImage()(masks[i, 0])
#             mask.set_frame(frame)
#             write_image(mask, os.path.join(case_out_folder, 'mask.nii.gz'))

#         if outputs is not None:
#             output = ToImage()(outputs[i, 0].data)
#             output.set_frame(frame)
#             write_image(output, os.path.join(case_out_folder, 'output.nii.gz'))
            

def validate(net, opt, val_loader, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality, max_vscore, last_save_epoch):
    val_iter = iter(val_loader)
    last_val_epoch = epoch_idx  
    inferer = SlidingWindowInferer(roi_size=cfg.dataset.crop_size, sw_batch_size=2, overlap=0.25, cval=-4) # 训练padding-1024，归一化后约-4         
    with torch.no_grad():
        net.eval()
        vscore_sum = 0
        for _ in range(len(val_loader)):
            crops, mask = next(val_iter)
            crops, mask = crops.cuda(), mask.cuda()
            print(crops.shape)
            print(mask.shape)
            output = inferer(inputs=crops, network=net)
            val_score =  binary_dice_score(output, mask)
            # val_score = multiclass_dice_score(output, mask)
            print('val score', val_score)
            vscore_sum+=val_score.item()
        vscore_mean = vscore_sum/len(val_loader)
        print('mean val score:', vscore_mean)
        if vscore_mean > max_vscore:
            max_vscore = vscore_mean
            best_epoch = epoch_idx
            print('better val score {} at epoch {}'.format(vscore_mean, epoch_idx))
            info = {}
            l1 = [best_epoch]
            l2 = [max_vscore]
            info['best epoch'] = l1
            info['val score'] = l2
            out = pd.DataFrame(info)
            out.to_csv(cfg.general.save_dir+'validation.csv')
            last_save_epoch = epoch_idx
            save_checkpoint(net, opt, last_save_epoch, batch_idx, cfg, config_file, max_stride, num_modality, max_vscore, last_val_epoch)
    return last_save_epoch, last_val_epoch, max_vscore


def train(config_file, msg_queue=None):
    """ volumetric segmentation training engine

    :param config_file: the input configuration file
    :param msg_queue: message queue to export runtime message to integrated system
    :return: None
    """
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

    # convert to absolute path since cfg uses relative path
    root_dir = os.path.dirname(config_file)
    cfg.general.imseg_list = os.path.join(root_dir, cfg.general.imseg_list)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    # clean the existing folder if not continue training
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write("Found non-empty save dir.\n"
                         "Type 'yes' to delete, 'no' to continue: ")
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'vseg')

    # enable CUDNN
    cudnn.benchmark = True

    transform = create_transform(cfg.dataset)
    # initialize dataset
    dataset = SegmentationDataset(
        data_file=cfg.general.data_file,
        transform=transform
    )
    val_dataset = Val_dataset(data_file=cfg.general.val_list, transform=transform)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    # val_sampler = EpochConcateSampler(val_dataset, 1)

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_dataset, batch_size=1,
                             num_workers=cfg.train.num_threads, pin_memory=True, worker_init_fn=worker_init, shuffle=False)
    # define network
    gpu_ids = list(range(cfg.general.num_gpus))
    net_module = create_net(cfg)

    use_ds = cfg.net.get('use_ds', False)
    if use_ds:
        assert cfg.net.name == 'vbnet', 'only vbnet support for deep supervision training'
        net = net_module.SegmentationNet(dataset.num_modality(), cfg.dataset.num_classes, use_ds)
        net_numpool = int(math.log(max(net.max_stride()), 2))  # number of network scales
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # we don't use the lowest 1 output. Normalize weights so that they sum to 1
        loss_mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~loss_mask] = 0
        weights = weights[::-1] / weights.sum()
    else:
        net = net_module.SegmentationNet(dataset.num_modality(), cfg.dataset.num_classes)
    print("num modality ",dataset.num_modality())
    max_stride = net.max_stride()
    # net_module.vnet_kaiming_init(net)
    net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda() 
    
    assert np.all(np.array(cfg.dataset.crop_size) % np.array(max_stride) == 0), 'crop size not divisible by max stride'

    # define loss function
    
    loss_func = create_loss(cfg)

    # training optimizer
    opt = getattr(torch.optim, cfg.train.optimizer.name)(
        [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': cfg.train.lr}],
        lr=cfg.train.lr, **cfg.train.optimizer.params
    )

    # load checkpoint if resume epoch > 0
        
    if cfg.general.partial == True:
        print('partial load')
        last_save_epoch, batch_start = load_checkpoint_partial(cfg.general.resume_epoch, net, cfg.general.load_dir)
        last_val_epoch = -1
        max_vscore = -float('inf')
    else:
        # load checkpoint if resume epoch > 0
        if cfg.general.resume_epoch >= 0:
            print(f'complete load from epoch{cfg.general.resume_epoch}')
            last_save_epoch, batch_start, last_val_epoch, max_vscore = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir)
        else:
            print('train from scratch')
            last_save_epoch, batch_start = 0, 0
            last_val_epoch = -1
            max_vscore = -float('inf')

    # opt.param_groups[0]['lr'] = 1e-3
    # print(opt.param_groups[0]['lr'])
    scheduler = getattr(torch.optim.lr_scheduler, cfg.train.lr_scheduler.name)(
        optimizer=opt, **cfg.train.lr_scheduler.params)
    

    batch_idx = batch_start
    if cfg.general.clear_start_epoch:
        batch_idx = 0
    data_iter = iter(data_loader)

    
    # loop over batches
    for i in range(len(data_loader)):

        begin_t = time.time()

        crops, masks, frames, filenames = next(data_iter)
        print('input shape:', crops.shape)

        # save training crops for visualization
        # if cfg.debug.save_inputs:
        #     batch_size = crops.size(0)
        #     save_intermediate_results(list(range(batch_size)), crops, masks, None, frames, filenames,
        #                               cfg.general.save_dir)

        crops, masks = crops.cuda(), masks.cuda()

        if torch.isnan(crops).any() == True:
            print('invalid net inputs')
        if torch.isnan(masks).any() == True:
            print('invalid masks')

        # clear previous gradients
        opt.zero_grad()
        net.train()

        # network forward
        outputs = net(crops)
        if torch.isnan(outputs).any() == True:
            print('invalid outs')

        # the epoch idx of model
        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)

        if use_ds:
            assert isinstance(outputs, list), "The deep supervision network output should be of the list type."
            train_loss = 0
            if cfg.loss.name == 'BoundaryLoss':
                for output, weight in zip(outputs, weights):
                    train_loss += loss_func(output, masks)[0] * weight
            else:
                for output, weight in zip(outputs, weights):
                    train_loss += loss_func(output, masks) * weight
        else:
            if cfg.loss.name == 'BoundaryLoss':
                train_loss, _ = loss_func(outputs, masks)
            else:
                train_loss = loss_func(outputs, masks)

        print('loss: ',train_loss)
        # backward propagation
        train_loss.backward()

        # update weights
        opt.step()
        

        if epoch_idx != scheduler.last_epoch:
            scheduler.step(epoch=epoch_idx)
        print('lr', opt.param_groups[0]['lr'])

        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)
        if msg_queue is not None:
            msg_queue.put(msg)
        
        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            train_loss_plot_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
            plot_loss(log_file, train_loss_plot_file, name='train_loss',
                      display='Training Loss ({})'.format(cfg.loss.name))

        # validation
        if epoch_idx != 0 and (epoch_idx % cfg.train.val_epoch == 0) and epoch_idx != last_val_epoch:
            print('val round')
            last_save_epoch, last_val_epoch, max_vscore = validate(net, opt, val_loader, epoch_idx,  batch_idx, cfg, config_file, max_stride,\
                                                                    dataset.num_modality(), max_vscore, last_save_epoch)
            
        # save checkpoints
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0) and epoch_idx != last_save_epoch:
            last_save_epoch = epoch_idx
            save_checkpoint(net, opt, last_save_epoch, batch_idx, cfg, config_file, max_stride, dataset.num_modality(), max_vscore, last_val_epoch)
                      

def main():
    long_description = "UII Segmentation3d Train Engine"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default='/root/projects/aortic_dissection/configs/AO_general_config.yaml',
                        help='volumetric segmentation3d train config file')
    parser.add_argument('-p', '--preprocess', default=False, help='whether to do data preprocess analysis')
    parser.add_argument('-s', '--scale', default='coarse', help='set to \'coarse\' or \'fine\', '
                        'scale used for training when data preprocess analysis result contains multiple scales')
    args = parser.parse_args()

    config_file = args.input
    # scale = args.scale

    train(config_file)

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
    main()
