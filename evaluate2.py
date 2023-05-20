from __future__ import absolute_import, division, print_function

import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

import datasets
from metrics import Evaluator
from networks import CDF, CDSFW, CDSFS, CDSFSU, CDFE
from networks.midas_small import MidasNet_small
from saver import Saver
from networks.E2P import equi2pers
import torch.nn.functional as F

#/media/lyb/CE7258D87258C73D/linux/github/DataSet/Matt
parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="/media/lyb/CE7258D87258C73D/linux/github/DataSet/Matt", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="matterport3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to evaluate on.")

parser.add_argument("--load_weights_dir", type=str, help="folder of model to load")
parser.add_argument("--network",default="CDF", type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=10, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", action="store_true", help="if set, save the depth maps and point clouds")

settings = parser.parse_args()

def main():

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model.pth")
    pretrained_dict = torch.load(model_path)

    # data
    datasets_dict = {"3d60": datasets.ThreeD60,
                     "panosuncg": datasets.PanoSunCG,
                     "stanford2d3d": datasets.Stanford2D3D,
                     "matterport3d": datasets.Matterport3D}
    dataset = datasets_dict[settings.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    test_file_list = fpath.format(settings.dataset, "test")

    test_dataset = dataset(settings.data_path, test_file_list,
                           pretrained_dict['height'], pretrained_dict['width'], is_training=False)
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net = { "CDF": CDF,
            "CDFE": CDFE,
            "CDSFW": CDSFW,
            "CDSFS": CDSFS,
            "CDSFSU": CDSFSU}

    model = Net[settings.network](pretrained_dict['layers'], pretrained_dict['height'], pretrained_dict['width'])
    model.to(device)
    model_state_dict = model.state_dict()

    new_pre ={}
    for k,v in pretrained_dict.items():
        name = k[7:]
        if name in model_state_dict:
            new_pre[name] = v

    model.load_state_dict(new_pre)
    model.eval()

    midas_model_path = os.getcwd() + "/networks/midas/midas_v21_small-70d6b9c8.pt"
    depthinit = MidasNet_small(midas_model_path)
    depthinit.to(device)
    depthinit.eval()

    evaluator = Evaluator(settings.median_align)
    evaluator.reset_eval_metrics()
    saver = Saver(load_weights_folder)
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    idx = 1
    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["normalized_rgb"].to(device)
            equi_rgb = inputs["rgb"].to(device)
            roll_idx = inputs["roll_idx"].to(device)
            flip = inputs["flip"].to(device)
            # depth_patch = inputs["depth_patch"].to(device)

            rgb_patch, _, _, _ = equi2pers(equi_rgb, 105, 3, patch_size=256)
            patch = torch.squeeze(torch.cat(torch.split(rgb_patch, 1, dim=-1), dim=0), dim=-1)  # bs*18, 3, 128, 128
            outputs = torch.squeeze(torch.cat(torch.split(depthinit(patch), 1, dim=0), dim=-1), dim=0)
            depth_min = outputs.min()
            depth_max = outputs.max()
            max_val = (2 ** (8 * 1)) - 1
            depth_patch = (1 - (max_val * (outputs - depth_min) / (depth_max - depth_min)).int() / max_val) * 10.0
            depth_path = torch.unsqueeze(depth_patch, dim=0)
            outputs = model(equi_inputs, depth_patch, roll_idx, flip)

            pred_depth = outputs["pred_depth"].detach().cpu()

            sky_pre = outputs['skyseg'].detach().cpu()
            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]
            #
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])
            if settings.save_samples:
                saver.save_samples(inputs["rgb"],  pred_depth, sky_pre=sky_pre)

    evaluator.print(load_weights_folder)


if __name__ == "__main__":
    main()
