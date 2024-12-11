import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib

# from md.image3d.python.image3d import Image3d
# import md.image3d.python.image3d_io as cio
# import md.image3d.python.image3d_tools as ctools
# from md.mdpytorch.utils.tensor_tools import ToTensor
# import md.image3d.python.image3d_tools as imtools
# from md_segmentation3d.utils.vseg_helpers import random_crop_with_data_augmentation, multitask_random_crop_with_data_augmentation
from tqdm import tqdm
from monai.transforms import Rotate, LoadImaged, ToTensord
import time
import nibabel as nib


def read_train_csv(csv_file):
    """ read multi-modality csv file

    :param csv_file: csv file path
    :return: a list of image path list, list of segmentation paths, list of sampling segmentation paths
    """
    im_list, seg_list, sampling_list = [], [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(reader)

        has_sampling = headers[-1] == 'sampling'
        if not has_sampling:
            # if no sampling paths are specified
            assert headers[-1] == 'segmentation'
            num_modality = len(headers) - 1
        else:
            # if sampling paths are specified
            assert headers[-2] == 'segmentation'
            num_modality = len(headers) - 2

        for i in range(num_modality):
            assert headers[i] == 'image{}'.format(i+1)

        for line in reader:
            for path in line:
                assert os.path.exists(path) or path == '', 'file not exist: {}'.format(path)

            if not has_sampling:
                im_list.append(line[:-1])
                seg_list.append(line[-1])
                sampling_list.append(line[-1])
            else:
                im_list.append(line[:-2])
                seg_list.append(line[-2])
                if line[-1].strip() != '':
                    sampling_list.append(line[-1])
                else:
                    sampling_list.append(line[-2])

    return im_list, seg_list, sampling_list

def multiMask_read_train_csv(csv_file, num_mask):
    """ read multi-modality csv file

    :param csv_file: csv file path
    :return: a list of image path list, list of segmentation paths, list of sampling segmentation paths
    """
    im_list, seg_list, sampling_list = [], [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(reader)

        has_sampling = headers[-1] == 'sampling'
        if not has_sampling:
            # if no sampling paths are specified
            assert headers[-1] == 'segmentation'
            num_modality = len(headers) - num_mask
        else:
            # if sampling paths are specified
            assert headers[-2] == 'segmentation'
            num_modality = len(headers) - num_mask - 1

        for i in range(num_modality):
            assert headers[i] == 'image{}'.format(i+1)

        for line in reader:
            for path in line:
                assert os.path.exists(path) or path == '', 'file not exist: {}'.format(path)

            if not has_sampling:
                im_list.append(line[:num_modality])
                seg_list.append(line[num_modality:])
                sampling_list.append(line[-1])
            else:
                im_list.append(line[:-2])
                seg_list.append(line[-2])
                if line[-1].strip() != '':
                    sampling_list.append(line[-1])
                else:
                    sampling_list.append(line[-2])
    assert len(seg_list[0]) == num_mask
    return im_list, seg_list, sampling_list

class SegmentationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, data_file:str, transform=None):
        assert data_file.endswith(".csv"), "Only support csv input file"
        im_list, seg_list, sampling_list = read_train_csv(data_file)
        self.im_list = im_list
        self.seg_list = seg_list
        self.sampling_list = sampling_list
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        img_data = nib.load(self.im_list[idx], dtype=np.float32).get_fdata()
        seg_data = nib.load(self.seg_list[idx], dtype=np.float32).get_fdata()
        
        # 将numpy数组转换为字典，以便与MONAI的转换兼容
        sample = {"image": img_data, "mask":seg_data}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class Val_dataset(Dataset):
    # TODO
    def __init__(self, data_file, transforms):
        super().__init__()
        
        self.im_list, self.seg_list, self.sampling_list = read_train_csv(data_file)
        self.transforms = transforms
        print("val set init done")
    def __len__(self):
        if isinstance(self.im_list, list):
            return len(self.im_list)
        return None
    def __getitem__(self, idx):
        img_paths, seg_path = self.im_list[idx], self.seg_list[idx]
        seg = LoadImaged(seg_path)
        
        images=[]
        for idx, impath in enumerate(img_paths):
            img = LoadImaged(impath)
            images.append(img)
        sample = {"image": images, "mask":seg}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

# class old_Val_dataset(Dataset):
#     def __init__(self, data_csv, normalizers):
#         super().__init__()
        
#         csv = pd.read_csv(data_csv)
#         heads = list(csv.columns)
#         data = []
#         labels = []
#         for row in range(csv.shape[0]):
#             input_list = []
#             for col in range(csv.shape[1] - 1): #last col 'segmentation', first n-1 cols 'image1' ,'image2'...
#                 input = cio.read_image(csv.loc[row, heads[col]])
#                 if input.pixel_type() != np.float32:
#                     input = input.deep_copy()
#                     input.cast(np.float32)
#                 normalizers[col](input)
#                 input_list.append(input)
#             vals = ToTensor()(input_list)
#             data.append(vals)
#         for row in range(csv.shape[0]):
#             assert heads[-1] == 'segmentation'
#             label = cio.read_image(csv.loc[row, heads[-1]], dtype=np.float32)
#             label = ToTensor()(label)
#             labels.append(label)
        
#         self.data = data
#         self.label = labels
#         # print(self.data[0].shape)
#         # print(self.label[0].shape)
#         # assert self.data.shape == self.label.shape
#         print("val set init done")
#     def __len__(self):
#         if isinstance(self.data, list):
#             return len(self.data)
#         else:
#             return self.data.shape[0]
    
#     def __getitem__(self, idx):
#         return self.data[idx], self.label[idx]



# class TrackingSegmentationDataset(Dataset):
#     """ training data set for volumetric segmentation """

#     def __init__(self, imlist_file, num_classes, spacing, crop_size, pad_t, default_values, sampling_method,
#                  interpolation, crop_normalizers, mixed_rate, box_sample_padding, aug_config):
#         """ constructor
#         :param imlist_file: image-segmentation list file
#         :param num_classes:         the number of classes
#         :param spacing:             the resolution, e.g., [1, 1, 1]
#         :param crop_size:           crop size, e.g., [96, 96, 96]
#         :param pad_t:               re-sample padding type, 0 for zero-padding, 1 for edge-padding
#         :param default_values:      default padding value list, e.g.,[0]
#         :param sampling_method:     'GLOBAL', 'MASK'
#         :param interpolation:       'LINEAR' for linear interpolation, 'NN' for nearest neighbor
#         :param crop_normalizers:    used to normalize the image crops, one for one image modality
#         :param mixed_rate:          param for mixed sample
#         :param box_sample_padding:  param for box sample and mixed sample
#         :param aug_prob:            the probability to apply data augmentation
#         :param shift_config:        the config of shift augmentation
#         :param rot_config:          the config of rotate augmentation
#         :param flip_config:         the config of flip augmentation
#         :param scale_config:        the config of scale augmentation
#         :param cover_config:        the config of cover augmentation
#         """
#         if imlist_file.endswith('txt'):
#             self.im_list, self.seg_list, self.sampling_list = read_train_txt(imlist_file)
#         elif imlist_file.endswith('csv'):
#             self.im_list, self.seg_list, self.sampling_list = read_train_csv(imlist_file)
#         else:
#             raise ValueError('imseg_list must either be a txt file or a csv file')

#         self.num_classes = num_classes
#         self.pad_t = pad_t
#         self.default_values = default_values
#         self.aug_config = aug_config

#         self.spacing = np.array(spacing, dtype=np.double)
#         assert self.spacing.size == 3, 'only 3-element of spacing is supported'

#         self.crop_size = np.array(crop_size, dtype=np.int32)
#         assert self.crop_size.size == 3, 'only 3-element of crop size is supported'

#         assert sampling_method in ('GLOBAL', 'MASK', 'MIXED', 'centerline', 'BOX'), 'sampling_method must either be GLOBAL, ' \
#                                                                       'MASK , BOX or MIXED'
#         self.sampling_method = sampling_method

#         assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
#         self.interpolation = interpolation

#         assert isinstance(crop_normalizers, list), 'crop normalizers must be a list'
#         self.crop_normalizers = crop_normalizers

#         assert 0 <= mixed_rate <= 1, 'mixed rate must be [0, 1]'
#         self.mixed_rate = mixed_rate

#         assert len(box_sample_padding) == 6, \
#             'box_sample_padding must be in format [x_min_pad, y_min_pad, z_min_pad, x_max_pad, y_max_pad, z_max_pad]'
#         self.box_sample_padding = np.array(box_sample_padding)

#         # self.aug_prob = aug_prob
#         # self.shift_config = shift_config
#         # self.rot_config = rot_config
#         # self.flip_config = flip_config
#         # self.scale_config = scale_config
#         # self.cover_config = cover_config

#     def __len__(self):
#         """ get the number of images in this data set """
#         return len(self.im_list)

#     def num_modality(self):
#         """ get the number of input image modalities """
#         return len(self.im_list[0])

#     def global_sample(self, image):
#         """ random sample a position in the image
#         :param image: a image3d object
#         :return: a position in world coordinate
#         """
#         assert isinstance(image, Image3d)
#         min_box, im_size_mm = image.world_box_full()
#         crop_size_mm = self.crop_size * self.spacing

#         sp = np.array(min_box, dtype=np.double)
#         for i in range(3):
#             if im_size_mm[i] > crop_size_mm[i]:
#                 sp[i] = min_box[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
#         center = sp + crop_size_mm / 2
#         return center

#     def centerline_sample(self, seg):
#         """random sample crop center from aortic centerline

#         Args:
#             seg (mask): seg_mask
#         """
#         im_size = seg.size()
#         len_z = im_size[2]
        
#         do_aug = np.random.choice([False, True], p=[0.7, 0.3])
#         if do_aug:
#             tmp_center = ctools.random_voxels_multi(seg, 1, [1])
#             if len(tmp_center) > 0:
#                 random_z = tmp_center[0][2]
#             else:
#                 random_z = np.random.randint(0, len_z)
#         else:
#             random_z = 0
            
#         center = np.array([int(im_size[0] / 2), int(im_size[1] / 2), random_z])
        
#         center = seg.voxel_to_world(center)
        
#         return center
    
#     def box_sample(self, seg):
#         labels = imtools.count_labels(seg)
#         if len(labels) == 0:
#             # if no segmentation
#             center = self.global_sample(seg)
#         else:
#             # if segmentation exists
#             minbox, maxbox = imtools.bounding_box_voxel(seg, np.min(labels), np.max(labels))
#             world_max, world_min = seg.voxel_to_world(minbox), seg.voxel_to_world(maxbox)
#             world_min -= self.box_sample_padding[0:3]
#             world_max += self.box_sample_padding[3:6]
#             center = np.array(world_min, dtype=np.double)
#             for i in range(3):
#                 center[i] = np.random.uniform(0, 1) * (world_max[i] - world_min[i]) + world_min[i]
#         return center

#     def mask_sample(self, seg):
#         labels = imtools.count_labels(seg)

#         labels = [i for i in range(np.max(seg.view_numpy()))]

#         # labels = np.array([20])
#         if len(labels) == 0:
#             # if no segmentation
#             center = self.global_sample(seg)
#         else:
#             # if segmentation exists
#             center = ctools.random_voxels_multi(seg, 1, labels)
#             if len(center) > 0:
#                 center = seg.voxel_to_world(center[0])
#                 # print (ctools.get_pixel_value(seg, seg.world_to_voxel(center)))
#             else:
#                 center = self.global_sample(seg)
#         return center
    
#     def random_sample_mask(self, mask):
#         cneterz = int(self.crop_size[2] /2)
#         sign = [1, -1]
#         sign_idx = np.random.randint(0, 2)
#         sign = sign[sign_idx]
#         mask_np = mask.to_numpy()
#         # print('mask_np shape', mask_np.shape)
#         mask_np_mid = mask_np[cneterz-1, :]
#         # print('mask_np_mid shape', mask_np_mid.shape)
#         idxs = np.argwhere(mask_np_mid==1)
#         if len(idxs) == 0:
#             return mask.to_numpy()
#         random_point = np.random.randint(0, len(idxs))
#         start_voxel = np.append(cneterz-1, idxs[random_point]) #coordinate [z, x, y]
#         iternum = np.random.randint(5, 15)
#         num_each_step = 3
#         aug_direction = [[1,0,0],[-1,0,0],
#                             [0,1,0],[0,-1,0],
#                             [0,0,1],[0,0,-1]]
#         mask_np_aug = np.zeros_like(mask_np)
#         mask_np_aug[start_voxel[0], start_voxel[1], start_voxel[2]] = 1
#         for iteridx in range(1, iternum+1):
#             directions = []
#             for voxel_num in range(num_each_step):
#                 directions.append(np.random.randint(0,6))
#             idxs = np.argwhere(mask_np_aug == iteridx) # last round aug indices
#             for idx in idxs:
#                 for direction in directions:
#                     tmp_idx = idx + aug_direction[direction] # new aug indices based on last round
#                     tmp_idx = np.clip(tmp_idx, a_min=[0,0,0], a_max=list(np.array(mask_np.shape)-1)) # coordinate range check
#                     if sign == 1:
#                         mask_np_aug[tmp_idx[0], tmp_idx[1], tmp_idx[2]] = iteridx + 1
#                     else:
#                         if mask_np[tmp_idx[0], tmp_idx[1], tmp_idx[2]] != 1: # when generating FN, only on original foreground area
#                             continue
#                         else:
#                             mask_np_aug[tmp_idx[0], tmp_idx[1], tmp_idx[2]] = iteridx + 1
#         mask_np_aug = np.int8(mask_np_aug > 0) # set to 1
#         mask_np = mask_np + sign * mask_np_aug # affect original mask
#         mask_np = np.float32(mask_np > 0)

#         # mask_np[0:cneterz, :] = 0 # mask half to 0

#         return mask_np


#     def __getitem__(self, index):
#         """ get a training sample - image(s) and segmentation pair
#         :param index:  the sample index
#         :return cropped image, cropped mask, crop frame, case name
#         """
#         image_paths, seg_path, sampling_path = self.im_list[index], self.seg_list[index], self.sampling_list[index]

#         case_name = os.path.basename(os.path.dirname(image_paths[0]))
#         case_name += '_' + os.path.basename(image_paths[0])

#         seg = cio.read_image(seg_path, dtype=np.int16)

#         # sampling_path = seg_path.replace("Crop_Dissection_Ture_mask.nii.gz", "Crop_lumne_mask_centerline_index.nii.gz")

#         if seg_path == sampling_path:
#             sampling_seg = seg
#         else:
#             sampling_seg = cio.read_image(sampling_path, dtype=np.int16)

#         # sampling a crop center
#         if self.sampling_method == 'GLOBAL':
#             center = self.global_sample(sampling_seg)
#         elif self.sampling_method == 'MASK':
#             center = self.mask_sample(sampling_seg)
#         elif self.sampling_method == 'BOX':
#             center = self.box_sample(sampling_seg)
#         elif self.sampling_method == 'centerline':
#             center = self.centerline_sample(sampling_seg)
#         elif self.sampling_method == 'MIXED':
#             if np.random.uniform(0, 1) < self.mixed_rate:
#                 center = self.mask_sample(sampling_seg)
#             else:
#                 center = self.box_sample(sampling_seg)
#         else:
#             raise ValueError('Only GLOBAL, MASK, BOX and MIXED are supported as sampling_method')

#         # sample crop from image with data augmentation
#         images, seg = random_crop_with_data_augmentation(
#             image_paths, center, self.spacing, self.crop_size, seg,
#             self.aug_config, self.interpolation, pad_type=0, pad_values=self.default_values
#         )

#         half_ind = int(self.crop_size[2]/2)
        
#         # images0_np = np.flip(images[0].to_numpy(), 0)
#         # images[0].from_numpy(images0_np)
        
#         random_augment_prob = 0.5
#         zero_mask_prob = 0.3
#         do_aug = np.random.choice([False, True], p=[1 - random_augment_prob, random_augment_prob]) if 0 <= random_augment_prob <= 1 else False
#         is_zero_mask = np.random.choice([False, True], p=[1 - zero_mask_prob, zero_mask_prob]) if 0 <= zero_mask_prob <= 1 else False   

#         # seg_np = seg.to_numpy()
#         # # seg_np[int(half_ind+5):,:,:] = 0
#         # seg.from_numpy(seg_np)

#         #images[1] part mask

#         images_1_np = images[1].to_numpy()
#         images_1_np[0:int(half_ind), :] = 0
#         images[1].from_numpy(images_1_np)

#         if is_zero_mask:
#             images[1].from_numpy(np.zeros_like(images_1_np))
#         elif do_aug:
#             images1_np = self.random_sample_mask(images[1])
#             images[1].from_numpy(images1_np)

#         # train with partial zero prev mask
        
#         # random_augment_prob = 0.2
#         # do_aug = np.random.choice([False, True], p=[1 - random_augment_prob, random_augment_prob]) if 0 <= random_augment_prob <= 1 else False
#         # if do_aug:
#         #     iter_num = np.random.randint(2, 6)
#         #     morph.imerode(images[1], iter_num, 1)
        
#         # normalize crop image
#         for idx in range(len(images)):
#             if self.crop_normalizers[idx] is not None:
#                 self.crop_normalizers[idx](images[idx])

#         # set labels of not interest to zero
#         ctools.convert_labels_outside_to_zero(seg, 1, self.num_classes - 1)

#         # image frame
#         frame = images[0].frame().to_numpy()

#         # convert to tensors
#         im = ToTensor()(images)
#         seg = ToTensor()(seg)

#         return im, seg, frame, case_name
