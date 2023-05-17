import os

import numpy as np
import cv2
#import open3d as o3d
import torch.nn.functional as F

from scipy.cluster.vq import *
from depth2normal import depth2normal
import time
from skimage.measure import label
# from plane import plane

def gradient_x(img):
    # Pad input to keep output size consistent
    img = np.pad(img, ((0, 0), (0, 1)), mode="reflect")
    gx = img[:, :-1] - img[ :, 1:]  # NCHW
    return gx

def gradient_y(img):
    # Pad input to keep output size consistent
    img = np.pad(img, ((0, 1), (0, 0)), mode="edge")
    gy = img[:-1, :] - img[1:, :]  # NCHW
    return gy

def Gradient(img, threshold):
    img_dx = gradient_x(img)
    img_dy = gradient_y(img)
    diff = np.abs(img_dx) + np.abs(img_dy)
    diff = diff
    edge = np.where(diff > threshold, 1.0, 0.0)
    return edge

def connect_domain(img, threshold):
    labeled_mask, num = label(img, background=0, return_num=True)
    label_mask = np.zeros(img.shape[:])
    for h in range(1, num + 1):
        if np.sum(labeled_mask == h) > threshold:
            labeled = np.where(labeled_mask == h, 1, 0)
            label_mask +=labeled
    return label_mask

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def find_max_region(mask_sel):
    labeled_mask, num = label(mask_sel, neighbors=4, background=0, return_num=True)
    max_region_index = 0
    max_region=0
    for h in range(1, num + 1):
        label_num = np.where(labeled_mask == h, 1.0, 0.0)
        sum_label = np.sum(label_num)
        if sum_label > max_region:
            max_region = sum_label
            max_region_index = label_num

    return max_region_index

class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    # def save_as_point_cloud(self, depth, rgb, path, mask=None):
    #     h, w = depth.shape
    #     Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    #     Theta = np.repeat(Theta, w, axis=1)
    #     Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    #     Phi = -np.repeat(Phi, h, axis=0)
    #
    #     X = depth * np.sin(Theta) * np.sin(Phi)
    #     Y = depth * np.cos(Theta)
    #     Z = depth * np.sin(Theta) * np.cos(Phi)
    #
    #     if mask is None:
    #         X = X.flatten()
    #         Y = Y.flatten()
    #         Z = Z.flatten()
    #         R = rgb[:, :, 0].flatten()
    #         G = rgb[:, :, 1].flatten()
    #         B = rgb[:, :, 2].flatten()
    #     else:
    #         X = X[mask]
    #         Y = Y[mask]
    #         Z = Z[mask]
    #         R = rgb[:, :, 0][mask]
    #         G = rgb[:, :, 1][mask]
    #         B = rgb[:, :, 2][mask]
    #
    #     XYZ = np.stack([X, Y, Z], axis=1)
    #     RGB = np.stack([R, G, B], axis=1)
    #
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(XYZ)
    #     pcd.colors = o3d.utility.Vector3dVector(RGB)
    #     o3d.io.write_point_cloud(path, pcd)


    def save_samples(self, rgbs, pred_depths, skydepth=None, coarse_scene_structure=None, sky_pre=None, depth_masks=None):
        """
        Saves samples
        """

        rgbs = rgbs.cpu().numpy()

        depth_preds = pred_depths.cpu().numpy()
        # depth_masks = depth_masks.cpu().numpy()
        # skydepth = skydepth.cpu().numpy()

        mkdirs(self.save_dir)

        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1

            depth_preds[i][0][depth_preds[i][0] > 10.0] = 10.0

            sky_confi = np.where(sky_pre[i][0] > 0.49, 1.0, 0.0)

            num_sky = np.sum(sky_confi)
            if num_sky > 8000:

                #########################################################

                # HSV segmentation
                rgb_temp = (rgbs[i].transpose(1,2,0) * 255).astype(np.uint8)
                HSV_background = cv2.cvtColor(rgb_temp, cv2.COLOR_RGB2HSV)

                lower_blue = (78, 31, 46)
                higher_blue = (124, 255, 255)
                lower_white = (0, 0, 190)
                higher_white = (180, 46, 255)

                mask_b = cv2.inRange(HSV_background, lower_blue, higher_blue) / 255
                mask_w = cv2.inRange(HSV_background, lower_white, higher_white) / 255
    
                sky_confi_max = find_max_region(1-sky_confi)
                sky_confi = 1- sky_confi_max
                #
                mask = (np.where((mask_w+mask_b)>0.0, 1.0, 0.0) * sky_confi)
                
                dep = (1-mask)*depth_preds[i][0]
                # dep = depth_preds[i][0]


            else:
                dep = depth_preds[i][0]

            # dep = depth_preds[i][0]
            # dep[~depth_masks[i][0]]=0
            dep[:72, :] = 0
            dep[-72:, :] = 0
            max_val = (2 ** (8 * 1)) - 1
            preds_depth = max_val * (dep) / 10
            path = os.path.join(self.save_dir, '%04d' % (self.idx) + '.jpg')
            out = preds_depth
            out = cv2.applyColorMap(out.astype("uint8"), cv2.COLORMAP_INFERNO)
            cv2.imwrite(path, out)




