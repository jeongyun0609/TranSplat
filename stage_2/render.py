#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np

# def apply_colormap(gray, minmax=None, cmap=cv2.COLORMAP_PLASMA):
#     if type(gray) is not np.ndarray:
#         gray = gray.detach().cpu().numpy().astype(np.float32)
#     gray = gray.squeeze()
#     assert len(gray.shape) == 2
#     x = np.nan_to_num(gray)  # change nan to 0
#     if minmax is None:
#         mi = np.min(x)  # get minimum positive value
#         ma = np.max(x)
#     else:
#         mi, ma = minmax
#     x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
#     x = (255 * x).astype(np.uint8)
#     x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
#     x_ = T.ToTensor()(x_)  # (3, H, W)
#     return x_
def apply_colormap(gray, minmax=None, cmap=cv2.COLORMAP_PLASMA):
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    # print(mi, ma)
    # mi = 0
    # ma = 1000

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = np.clip(x,0,1)
    x = (255 * x).astype(np.uint8)
    # x_= x
    x_ = cv2.applyColorMap(x, cmap)
    return x_

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_metric")
    depth_norm_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_norm")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    diff_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diff")
    surf_path = os.path.join(model_path, name, "ours_{}".format(iteration), "surf")

    result_path = "result.txt"

    mseloss = torch.nn.L1Loss(reduction='sum')
    mseloss_ = torch.nn.MSELoss(reduction='sum')
    makedirs(render_path, exist_ok=True)
    makedirs(depth_norm_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(diff_path, exist_ok=True)
    makedirs(surf_path, exist_ok=True)

    diff = 0
    diff_ = 0
    pixels = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]
        depth_map = render_result["depth_map"]
        weight_map = render_result["weight_map"]
        surf_map = render_result["surf"]
        # A = torch.zeros_like(depth_map).to(torch.float64)
        gt = view.original_image[0:3, :, :]
        gt_depth = view.depth
        mask = gt_depth>0
        pixels += torch.sum(mask)
        diff +=mseloss(gt_depth[mask], depth_map[mask])
        diff_ +=mseloss_(gt_depth[mask], depth_map[mask])

        surf = view.surf[0:3, :, :]

        # A[mask] = torch.abs(gt_depth[mask] -  depth_map[mask])

        # diff += torch.sum(torch.mse(gt_depth[mask]-depth_map[mask]))/torch.sum(mask)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(surf, os.path.join(surf_path, '{0:05d}'.format(idx) + ".png"))
        depth_map_np = depth_map.detach().cpu().numpy().astype(np.float32)*1000
        depth_map_np = depth_map_np.astype(np.uint16)
        cv2.imwrite(os.path.join(depth_norm_path, '{0:05d}'.format(idx) + ".png"), apply_colormap(depth_map_np))
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_map_np)

        # torchvision.utils.save_image(apply_colormap(depth_map), os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(apply_colormap(A), os.path.join(diff_path, '{0:05d}'.format(idx) + ".png"))

    diff = diff / pixels
    diff_ = torch.sqrt(diff_ / pixels)
    print("L1norm:", diff)
    print("L2norm:", diff_)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)