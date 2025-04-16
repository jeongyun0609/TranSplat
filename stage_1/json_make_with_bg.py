import json
import os
import warnings
import cv2
import math
import numpy as np
import sys
import shutil
np.random.seed(seed=0)
import argparse
parser = argparse.ArgumentParser(description='R6D')
parser.add_argument('-i', '--index', required=True,
                    default=0, type=int,
                    help="index")  
parser.add_argument('--dataset', required=True,
                    default=None, type=str)  

args = parser.parse_args()
T_EE2Cam = np.array([
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
aabb_scale= 128
scale= 1
offset= [0.5,0.5,1]

out = {		"camera_angle_x": camera_angle_x,
			"camera_angle_y": camera_angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"k3": k3,
			"k4": k4,
			"p1": p1,
			"p2": p2,
			"is_fisheye": is_fisheye,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"aabb_scale": aabb_scale,
            "scale": scale,
            "offset": offset,
			"frames": []
		}
out_ = {
			"camera_angle_x": camera_angle_x,
			"camera_angle_y": camera_angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"k3": k3,
			"k4": k4,
			"p1": p1,
			"p2": p2,
			"is_fisheye": is_fisheye,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"aabb_scale": aabb_scale,
            "scale": scale,
            "offset": offset,
			"frames": []
		}

if "syn" in args.dataset:
    seq_index = f"syn_test_{args.index:02d}"
    test_path  = f"../stage_0/SurfEmb/data/bop/{args.dataset}/{seq_index}/train_pbr"
    image_folder = os.path.join(test_path,"rgb")
    depth_folder = os.path.join(test_path, 'depth')
    bg_folder = os.path.join(test_path, 'bg')
    surfemb_folder = os.path.join(test_path, 'SurfEmb', 'SurfEmb')
    remove_bg_folder = os.path.join(test_path, 'SurfEmb', 'remove_bg')
    with open(os.path.join(test_path,'scene_camera.json')) as f:
        cam_poses = json.load(f)

else:
    seq_index = f"seq_test_{args.index:02d}"
    test_path  = f"../stage_0/SurfEmb/data/bop/{args.dataset}/sequences/test/{seq_index}"
    image_folder = os.path.join(test_path,"rgb")
    depth_folder = os.path.join(test_path, 'depth')
    bg_folder = os.path.join(test_path, 'bg')
    surfemb_folder = os.path.join(test_path, 'SurfEmb', 'SurfEmb')
    remove_bg_folder = os.path.join(test_path, 'SurfEmb', 'remove_bg')
    with open(os.path.join(test_path,'World2EEs.json')) as f:
        cam_poses = json.load(f)


image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(".png")]


if "syn" in args.dataset:
    for idx, image_file in enumerate(image_files):
        image_name = image_file.split('.')[0]
        cam_rot = np.array(cam_poses[str(int(image_name))]['cam_R_w2c']).reshape(3,3)
        cam_tra = np.array(cam_poses[str(int(image_name))]['cam_t_w2c'])*0.001
        matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
        matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
        matrix_values = np.array(matrix_values)
        pose_openCV = np.linalg.inv(matrix_values)
        pose = pose_openCV
        image_path = os.path.join(image_folder,image_name+".png")
        depth_path = os.path.join(depth_folder,image_name+".png")
        surfemb_path = os.path.join(surfemb_folder,image_name+".png")
        bg_image_path = os.path.join(bg_folder,image_name+".png")
        remove_bg_image_path = os.path.join(remove_bg_folder,image_name+".png")
        pose_list = pose.tolist()
        new_frame = {
            "file_path": f"{image_path}",
            "depth_path": f"{depth_path}",
            "surfemb_path": f"{surfemb_path}",
            "remove_bg_path": f"{remove_bg_image_path}",
            "bg_path": f"{bg_image_path}",
            "transform_matrix": pose_list
        }
        if idx %2 == 0:
            out_["frames"].append(new_frame)

        elif idx%8 ==1:
            out["frames"].append(new_frame)

    with open(os.path.join(test_path,"transforms_val.json"), 'w') as f:
        json.dump(out_, f, indent="\t")
    with open(os.path.join(test_path,"transforms_test.json"), 'w') as f:
        json.dump(out_, f, indent="\t")
    with open(os.path.join(test_path,"transforms_train.json"), 'w') as f:
        json.dump(out, f, indent="\t")


else:
    for idx, image_file in enumerate(image_files):
        if idx%12!=0:
            continue
        image_name = image_file.split('.')[0]
        cam_rot = np.array(cam_poses[(image_file.split('_')[-1][0:6])]['rot']).reshape(3,3)
        cam_tra = np.array(cam_poses[(image_file.split('_')[-1][0:6])]['tra'])*0.001
        matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
        matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
        matrix_values = np.array(matrix_values)
        pose = np.matmul(matrix_values,T_EE2Cam)
        image_path = os.path.join(image_folder,image_name+".png")
        depth_path = os.path.join(depth_folder,image_name+".png")
        surfemb_path = os.path.join(surfemb_folder,image_name+".png")
        bg_image_path = os.path.join(bg_folder,image_name+".png")
        remove_bg_image_path = os.path.join(remove_bg_folder,image_name+".png")
        pose_list = pose.tolist()
        new_frame = {
            "file_path": f"{image_path}",
            "depth_path": f"{depth_path}",
            "surfemb_path": f"{surfemb_path}",
            "remove_bg_path": f"{remove_bg_image_path}",
            "bg_path": f"{bg_image_path}",
            "transform_matrix": pose_list
        }
        if idx %2 == 1:
            out_["frames"].append(new_frame)

        else:
            out["frames"].append(new_frame)

    with open(os.path.join(test_path,"transforms_val.json"), 'w') as f:
        json.dump(out_, f, indent="\t")
    with open(os.path.join(test_path,"transforms_test.json"), 'w') as f:
        json.dump(out_, f, indent="\t")
    with open(os.path.join(test_path,"transforms_train.json"), 'w') as f:
        json.dump(out, f, indent="\t")