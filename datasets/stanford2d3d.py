from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from .util import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        if is_training:
            self.rgb_depth_list = 10 * self.rgb_depth_list
        self.w = width
        self.h = height

        self.max_depth_meters = 8.0

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        # self.e2c = Equirec2Cube(self.h, self.w, self.h//2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        # rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb_name = self.root_dir+"/"+self.rgb_depth_list[idx][0]
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        # if self.is_training:
        #    sem = np.where(cv2.imread(rgb_name + "_sem.jpg", cv2.IMREAD_GRAYSCALE) > 0.0, 1.0, 0.0)
        #    sem = cv2.resize(sem, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # depth_patch = (1 - cv2.imread(rgb_name+"_patch_small.png", -1)/255)*self.max_depth_meters

        # depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        depth_name = self.root_dir+"/"+self.rgb_depth_list[idx][1]
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float)/512
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        roll_idx = 0
        flip = 0
        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w // 4)
            rgb = np.roll(rgb, roll_idx * 4, 1)
            # sem = np.roll(sem, roll_idx * 4, 1)
            gt_depth = np.roll(gt_depth, roll_idx * 4, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            flip = 1
            rgb = cv2.flip(rgb, 1)
            # sem = cv2.flip(sem, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb_name"] = self.rgb_depth_list[idx][0]
        inputs["dep_name"] = self.rgb_depth_list[idx][1]
        inputs["rgb"] = rgb
        inputs["roll_idx"] = roll_idx
        inputs["flip"] = flip
        # inputs["depth_patch"] = depth_patch
        inputs["normalized_rgb"] = self.normalize(aug_rgb)
        #if self.is_training:
        #    inputs["sem"] = torch.from_numpy(np.expand_dims(sem, axis=0))
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0.1) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))


        return inputs



