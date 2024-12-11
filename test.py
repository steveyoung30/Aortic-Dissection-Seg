# import monai.transforms
import nibabel
import pandas as pd
import os
# import shutil
from monai.transforms import Compose, EnsureChannelFirstd, RandCropByPosNegLabeld, Spacingd, SpatialPadd
from monai.transforms import ToTensord, NormalizeIntensityd, ThresholdIntensityd, ScaleIntensityRanged, ScaleIntensityRanged
import numpy as np

def generate_train_csv(file_path, out_dir):
    out_df = {"image1":[], "segmentation":[]}
    with open(file_path, "r") as f:
        for line in f:
            name = line.split("/")[-2]
            out_df["image1"].append(os.path.join("/root/autodl-tmp/train/CTA/", name, "CTA.nii.gz"))
            out_df["segmentation"].append(os.path.join("/root/autodl-tmp/train/CTA/", name, "ao_mask.nii.gz"))
    
    for img in out_df["image1"]:
        try:
            assert os.path.exists(img)
        except Exception as e:
            print(img)
    for seg in out_df["segmentation"]:
        try:
            assert os.path.exists(seg)
        except Exception as e:
            print(seg)
    out_df = pd.DataFrame(out_df)
    out_df.to_csv(out_dir, sep=",", index=False)

def test():
    crop_size = (128, 128, 256)
    transform = Compose([
    EnsureChannelFirstd(keys=["image", "mask"], channel_dim=0),  # 确保通道维度位于最前面
    Spacingd(
        keys=["image", "mask"],  # 指定需要重采样的键
        pixdim=(1, 1, 1),  # 目标体素间距
        mode=("trilinear", "nearest"),  # 图像和标签的不同插值模式
        align_corners=(True, True)  # 对齐角点参数
    ),
    SpatialPadd(keys=["image", "mask"], spatial_size=crop_size, mode='constant', constant_values=0),
    NormalizeIntensityd(keys=["image", "mask"], subtrahend=300, divisor=400), # Fixed Mean Std normalization
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
    image = nibabel.load("/root/autodl-tmp/test/CTA/AI01_0002_E0P0D0_1.0/CTA.nii.gz").get_fdata()
    image = image[np.newaxis, :]
    mask = nibabel.load("/root/autodl-tmp/test/CTA/AI01_0002_E0P0D0_1.0/ao_mask.nii.gz").get_fdata()
    mask = mask[np.newaxis, :]
    transformed = transform({"image": image, "mask": mask})
    print("exit")

if __name__ == "__main__":
    file_path = "/root/projects/Aortic/datasets/train/CTA_AO.txt"
    out_dir = "/root/projects/Aortic/datasets/train/CTA_AO.csv"
    # generate_train_csv(file_path, out_dir)
    test()