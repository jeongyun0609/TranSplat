import torch
from matplotlib import cm
import math
import trimesh

from PIL import Image
import json
import cv2
import numpy as np
import os
from pathlib import Path

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from distutils.dir_util import copy_tree
import pyrender
import argparse

parser = argparse.ArgumentParser(description='R6D')
parser.add_argument('--is_train', action='store_true')
args = parser.parse_args()
T_EE2Cam_R = np.array([
                    [ 0.03901212, -0.999075,    0.01808859,  0.07291089],
                    [ 0.9992336,   0.03906357,  0.0024998,   0.05048912],
                    [-0.00320409,  0.01797721,  0.99983326,  0.06090912],
                    [ 0.,          0.,          0.,          1.        ]
                    ], dtype=np.float64)

fl_x=603.7764783730764
fl_y= 604.631625241128
w= 640.0
h= 480.0
camera_angle_x=math.atan(w / (fl_x * 2)) * 2
camera_angle_y= angle_y = math.atan(h / (fl_y * 2)) * 2
k1=0
k2= 0
k3= 0
k4= 0
p1= 0
p2= 0
is_fisheye= False
cx= 329.259404556074
cy= 246.484798070861

camera_info = {
"cx":	cx,
"cy":	cy,
"depth_scale":	1,
"fx":	fl_x,
"fy":	fl_y,
"height":	h,
"width":	w
}

rot = np.matrix([
        [ 1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1]])
shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
extra_xf = np.matrix([
    [ -1, 0, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, 0, 1]])   
Axis_align = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],]
                        )
r = pyrender.OffscreenRenderer(w, h)

cam = pyrender.camera.IntrinsicsCamera(fl_x,
                                        fl_y, 
                                        cx, 
                                        cy, 
                                        znear=0.01, zfar=100.0, name=None)

def convert_transpose_to_bop(transpose_folder, bop_folder, is_train):
    if is_train:
        train_path = os.path.join(transpose_folder,"sequences","train")
        bop_train = os.path.join(bop_folder,"train")
    else:
        train_path = os.path.join(transpose_folder,"sequences","test")
        bop_train = os.path.join(bop_folder,"test")       

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(bop_train):
        os.mkdir(bop_train)    
    for folder_name in os.listdir(train_path):
        print(folder_name)
        folder_path = os.path.join(train_path, folder_name)
        scene_camera ={}
        scene_gt = {}
        scene_gt_info = {}
        if os.path.isdir(folder_path):
            mask_directory = os.path.join(folder_path, "cam_R","mask")
            Pose_json_dir = os.path.join(folder_path, "cam_R","pose")
            img_dir = os.path.join(folder_path, "cam_R","rgb")
            depth_dir = os.path.join(folder_path, "cam_R","depth")
            world2ee_path = os.path.join(folder_path, "World2EEs.json")
        bop_seq_path = os.path.join(bop_train,folder_name)
        if not os.path.exists(bop_seq_path):
            os.mkdir(bop_seq_path)

        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        img_files.sort()

        if not os.path.exists(os.path.join(bop_seq_path,"depth")):
            copy_tree(depth_dir, os.path.join(bop_seq_path,"depth"))
        if not os.path.exists(os.path.join(bop_seq_path,"mask")):
            copy_tree(mask_directory, os.path.join(bop_seq_path,"mask"))
        if not os.path.exists(os.path.join(bop_seq_path,"rgb")):
            copy_tree(img_dir, os.path.join(bop_seq_path,"rgb"))

        with open(world2ee_path) as f:
            cam_poses = json.load(f)

        for idx, image_file in enumerate(img_files):
            cam_rot = np.array(cam_poses[(image_file.split('_')[-1][0:6])]['rot']).reshape(3,3)
            cam_tra = np.array(cam_poses[(image_file.split('_')[-1][0:6])]['tra'])
            matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
            matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
            matrix_values = np.array(matrix_values)
            pose = np.matrix([
    [ 1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0, -1,  0],
    [ 0,  0,  0,  1]])@shift_coords @ extra_xf @ matrix_values @ T_EE2Cam_R @rot
            rotation_matrix = pose[:3, :3].reshape(-1).tolist()[0]
            translation_vector = np.squeeze(pose[:3, 3]).tolist()[0]
            scene_camera[f"{idx+1}"]={
                "cam_K": [fl_x, 0.0, cx, 0.0, fl_y, cy, 0.0, 0.0, 1.0],
                "depth_scale": 1,
                "cam_R_w2c": rotation_matrix,
                "cam_t_w2c": translation_vector}
     
            with open(os.path.join(Pose_json_dir,image_file.split('_')[-1][0:6]+".json")) as f:
                obj_pose_json = json.load(f)

            scene_gt[f"{idx+1}"]=[]
            scene_gt_info[f"{idx+1}"]=[]

            for obj_id, pose_info in obj_pose_json.items():
                scene_gt[f"{idx+1}"].append(
                {"cam_R_m2c": pose_info['rot'],
                "cam_t_m2c": pose_info['tra'],
                "obj_id": pose_info['obj_id'],
                "obj_name": pose_info['obj_name']
                }
                )
                mask = cv2.imread(os.path.join(mask_directory,pose_info['mask_img']))
                px_count_visib = np.sum(mask>0)
                scene_gt_info[f"{idx+1}"].append(
                {"bbox_obj": pose_info['bb'],
                "bbox_visib": pose_info['bb'],
                "px_count_valid": str(px_count_visib/3),
                "px_count_visib": str(px_count_visib/3),
                "visib_fract": 1.0
                }
                )

        with open(os.path.join(bop_seq_path,"scene_camera.json"), 'w') as jsonfile:
            json.dump(scene_camera, jsonfile)

        with open(os.path.join(bop_seq_path,"scene_gt.json"), 'w') as jsonfile:
            json.dump(scene_gt, jsonfile)
        with open(os.path.join(bop_seq_path,"scene_gt_info.json"), 'w') as jsonfile:
            json.dump(scene_gt_info, jsonfile)

transpose_folder = Path('data/bop/TRansPose/')
bop_folder = Path('data/bop/TRansPose/')
is_train = args.is_train
convert_transpose_to_bop(transpose_folder, bop_folder, is_train)
