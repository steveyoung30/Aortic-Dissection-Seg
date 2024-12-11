# import monai.transforms
import pandas as pd
import os
import shutil
# import nibabel as nb
# import monai
# import time
# import importlib
# from tqdm import tqdm
# import yaml
# import easydict as edict
# import numpy as np

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
    old = pd.read_csv("/root/projects/Aortic/datasets/train/CTA_AO.csv")
    sampled = old.sample(n=10, random_state=2, replace=False)
    sampled.to_csv("/root/projects/Aortic/datasets/train/sub10_CTA_AO.csv", sep=",", index=False)

if __name__ == "__main__":
    file_path = "/root/projects/Aortic/datasets/train/CTA_AO.txt"
    out_dir = "/root/projects/Aortic/datasets/train/CTA_AO.csv"
    # generate_train_csv(file_path, out_dir)
    test()