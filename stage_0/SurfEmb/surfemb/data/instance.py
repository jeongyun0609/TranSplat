import json
from pathlib import Path
from typing import Sequence
import warnings
import numpy as np
from tqdm import tqdm
import torch.utils.data
import os
from .config import DatasetConfig

class BopInstanceDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, pbr: bool, test: bool, cfg: DatasetConfig,
            obj_ids: Sequence[int],
            scene_ids=None, min_visib_fract=0.2, min_px_count_visib=2048,
            auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True,
    ):
        self.pbr, self.test, self.cfg = pbr, test, cfg
        if pbr:
            assert not test
            self.data_folder1 = dataset_root / 'train_pbr_1/train_pbr'
            self.data_folder2 = dataset_root / 'train_pbr_2/train_pbr'
            self.data_folder3 = dataset_root / 'train_pbr_3/train_pbr'
            self.img_folder = 'rgb'
            self.depth_folder = 'depth'
            self.img_ext = 'png'
            self.depth_ext = 'png'
        else:
            self.data_folder = dataset_root / (cfg.test_folder if test else cfg.train_folder)
            self.img_folder = cfg.img_folder
            self.depth_folder = cfg.depth_folder
            self.img_ext = cfg.img_ext
            self.depth_ext = cfg.depth_ext

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        with open("data/bop/TRansPose/class_index.json", "r") as st_json:
            class_idxs = json.load(st_json)
 
        self.instances = []

        if pbr:
            if scene_ids is None:
                scene_ids1 = sorted([(os.fspath(self.data_folder1)+"/"+p.name) for p in self.data_folder1.glob('*')])
                # scene_ids2 = sorted([(os.fspath(self.data_folder2)+"/"+p.name) for p in self.data_folder2.glob('*')])
                # scene_ids3 = sorted([(os.fspath(self.data_folder3)+"/"+p.name) for p in self.data_folder3.glob('*')])
                scene_ids = scene_ids1
                # scene_ids = scene_ids1+scene_ids2+scene_ids3
        else:
            if scene_ids is None:
                scene_ids = sorted([(os.fspath(self.data_folder)+"/"+p.name) for p in self.data_folder.glob('*')])

        with open("data/bop/TRansPose/object_label.json", "r") as st_json:
            self.obj_key = json.load(st_json)

        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_id_ = scene_id
            scene_folder = Path(scene_id)
            scene_gt = json.load((scene_folder / 'scene_gt.json').open())
            scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())
            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]['cam_K']).reshape((3, 3)).copy()
                cam_R_w2c = np.array(scene_camera[img_id]['cam_R_w2c']).reshape(3, 3)
                cam_t_w2c = np.array(scene_camera[img_id]['cam_t_w2c']).reshape(3, 1)
                world_pose = np.concatenate((
                    np.concatenate((cam_R_w2c, cam_t_w2c), axis=1),
                    [[0, 0, 0, 1]],
                ))
                for pose_idx, pose in enumerate(poses):
                    num_obj = len(poses)
                    if pbr:
                        obj_id = self.obj_key[pose['obj_id']]
                        if "N" in pose['obj_id']:
                            continue
                    else:
                        obj_id = pose['obj_id']
                        if "N" in pose['obj_name']:
                            continue
                    pose_info = img_info[pose_idx]
                    if float(pose_info['visib_fract']) < min_visib_fract:
                        continue
                    if float(pose_info['px_count_visib']) < min_px_count_visib:
                        continue
                    bbox_visib = pose_info['bbox_visib']
                    bbox_obj = pose_info['bbox_obj']

                    cam_R_obj = np.array(pose['cam_R_m2c']).reshape(3, 3)
                    cam_t_obj = np.array(pose['cam_t_m2c']).reshape(3, 1)
                    matrix_values = np.concatenate((
                        np.concatenate((cam_R_obj, cam_t_obj), axis=1),
                        [[0, 0, 0, 1]],
                    ))
                    cam_R_obj = matrix_values[:3,:3]
                    cam_t_obj = matrix_values[:3,3].reshape(3,1)
                    if pbr:
                        self.instances.append(dict(
                            scene_id=scene_id_, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx,
                            bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id], world_pose = world_pose, num_obj = num_obj, box = bbox_visib, class_idx=class_idxs[pose['obj_id'][:8]]
                        ))
                    else:
                        self.instances.append(dict(
                            scene_id=scene_id_, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx+1,
                            bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id], world_pose = world_pose, num_obj = num_obj, box = bbox_visib, class_idx=class_idxs[pose['obj_name'][:8]]
                        ))             
        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance



class BopInferenceDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, pbr: bool, test: bool, cfg: DatasetConfig,
            obj_ids: Sequence[int],
            scene_ids=None, min_visib_fract=0.2, min_px_count_visib=2048,
            auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True, data_folder = None
    ):
        self.pbr, self.test, self.cfg = pbr, test, cfg

        self.data_folder = dataset_root / data_folder

        self.img_folder = cfg.img_folder
        self.depth_folder = cfg.depth_folder
        self.img_ext = cfg.img_ext
        self.depth_ext = cfg.depth_ext

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        with open("data/bop/TRansPose/class_index.json", "r") as st_json:
            class_idxs = json.load(st_json)
 
        self.instances = []
        scene_ids = sorted([(os.fspath(self.data_folder)+"/"+p.name) for p in self.data_folder.glob('*')])
        with open("data/bop/TRansPose/object_label.json", "r") as st_json:
            self.obj_key = json.load(st_json)
        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_id_ = scene_id
            scene_folder = Path(scene_id)
            scene_gt = json.load((scene_folder / 'scene_gt.json').open())
            scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())
            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]['cam_K']).reshape((3, 3)).copy()
                cam_R_w2c = np.array(scene_camera[img_id]['cam_R_w2c']).reshape(3, 3)
                cam_t_w2c = np.array(scene_camera[img_id]['cam_t_w2c']).reshape(3, 1)
                world_pose = np.concatenate((
                    np.concatenate((cam_R_w2c, cam_t_w2c), axis=1),
                    [[0, 0, 0, 1]],
                ))
                for pose_idx, pose in enumerate(poses):
                    num_obj = len(poses)
                    if pbr:
                        obj_id = self.obj_key[pose['obj_id']]
                        if "N" in pose['obj_id']:
                            continue
                    else:
                        obj_id = pose['obj_id']
                        if "N" in pose['obj_name']:
                            continue
                    pose_info = img_info[pose_idx]
                    if float(pose_info['visib_fract']) < min_visib_fract:
                        continue
                    if float(pose_info['px_count_visib']) < min_px_count_visib:
                        continue
                    bbox_visib = pose_info['bbox_visib']
                    bbox_obj = pose_info['bbox_obj']

                    cam_R_obj = np.array(pose['cam_R_m2c']).reshape(3, 3)
                    cam_t_obj = np.array(pose['cam_t_m2c']).reshape(3, 1)
                    Axis_align = np.array([
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                    matrix_values = np.concatenate((
                        np.concatenate((cam_R_obj, cam_t_obj), axis=1),
                        [[0, 0, 0, 1]],
                    ))
                    matrix_values = Axis_align @ matrix_values
                    cam_R_obj = matrix_values[:3,:3]
                    cam_t_obj = matrix_values[:3,3].reshape(3,1)
                    if pbr:
                        self.instances.append(dict(
                            scene_id=scene_id_, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx,
                            bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id], world_pose = world_pose, num_obj = num_obj, box = bbox_visib, class_idx=class_idxs[pose['obj_id'][:8]]
                        ))
                    else:
                        self.instances.append(dict(
                            scene_id=scene_id_, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx+1,
                            bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id], world_pose = world_pose, num_obj = num_obj, box = bbox_visib, class_idx=class_idxs[pose['obj_name'][:8]]
                        ))             
        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


class BopInstanceAux:
    def init(self, dataset: BopInstanceDataset):
        pass

    def __call__(self, data: dict, dataset: BopInstanceDataset) -> dict:
        pass


def _main():
    from .config import tless
    for pbr, test in (True, False), (False, False), (False, True):
        print(f'pbr: {pbr}, test: {test}')
        data = BopInstanceDataset(dataset_root=Path('bop/tless'), pbr=pbr, test=test, cfg=tless, obj_ids=range(1, 121))
        print(len(data))


if __name__ == '__main__':
    _main()
