import argparse
from pathlib import Path
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
from .. import utils
from ..data import detector_crops
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from ..data import obj, instance
import json


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-poses', type=int, default=10000)
parser.add_argument('--max-pose-evaluations', type=int, default=1000)
parser.add_argument('--no-rotation-ensemble', dest='rotation_ensemble', action='store_false')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_folder', type=str, default='TRansPose_surfemb')
parser.add_argument('--data_folder', type=str, default='train')
parser.add_argument('--syn', action='store_true')


class save_emb_crop():
    def __init__(self):
        self.seq_index=[]
        self.obj_index = []
        self.out_list={}
        self.out_list_={}
        self.args = parser.parse_args()
        self.res_crop = self.args.res_crop
        self.device = torch.device(self.args.device)
        self.model_path = Path(self.args.model_path)
        assert self.model_path.is_file()
        self.model_name = self.model_path.name.split('.')[0]
        self.dataset = self.model_name.split('-')[0]
        self.debug = self.args.debug
        self.model = SurfaceEmbeddingModel.load_from_checkpoint(str(self.model_path)).eval().to(self.device)
        self.model.freeze()
        self.root = Path('data/bop') / self.dataset
        self.cfg = config[self.dataset]
        self.objs, self.obj_ids = load_objs(self.root / self.cfg.model_folder)
        assert len(self.obj_ids) > 0

        self.data = instance.BopInferenceDataset(
                    dataset_root=self.root, pbr= self.args.syn, test=False, cfg=self.cfg, obj_ids=self.obj_ids, auxs=self.model.get_infer_auxs(objs=self.objs, crop_res=self.res_crop, from_detections=False),
                    min_visib_fract=0.1, scene_ids=[1] if self.debug else None, data_folder = self.args.data_folder)
        
        self.rgb_interpolation=(cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)
        with open("data/bop/TRansPose/obj_key.json", "r") as st_json:
            self.obj_key = json.load(st_json)
        self.save_folder_path = os.path.join('data/bop', self.args.save_folder)
        os.makedirs(os.path.join('data/bop', self.args.save_folder), exist_ok=True)

    def infer(self, i, d):
        obj_idx = d['obj_idx']
        class_idx = d['class_idx']
        scene_id = d['scene_id'].split("/")[-1]
        obj_name = self.obj_key[str(obj_idx+1)]
        self.data_folder = self.args.data_folder
        os.makedirs(os.path.join(self.save_folder_path, self.data_folder), exist_ok=True)
        mask_folder = os.path.join(self.save_folder_path, f"{self.data_folder}/{scene_id}/mask_sub")
        rgb_folder = os.path.join(self.save_folder_path, f"{self.data_folder}/{scene_id}/rgb_sub")
        visible_mask_folder = os.path.join(self.save_folder_path, f"{self.data_folder}/{scene_id}/visible_sub")
        os.makedirs(os.path.join(self.save_folder_path, self.data_folder,scene_id), exist_ok=True)
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(visible_mask_folder, exist_ok=True)
        bbox = d['bbox_obj']
        if bbox[1]< 25 or bbox[3]+bbox[1]>455 or bbox[0] < 25 or bbox[2]+bbox[0] > 615:
            return 0
        mask = d['mask_visib_crop']
        mask_ = d['save_mask']
        coord_img = d['obj_coord']
        key_img = self.model.infer_mlp(coord_img[..., :3], class_idx)
        shape = key_img.shape[:-1]
        key_img_ = key_img.view(*shape, 3, -1).mean(dim=-1)
        key_img_ /= torch.abs(key_img_).max() + 1e-9
        key_img_.mul_(0.5).add_(0.5)
        if mask is not None:
            key_img_[~mask] = 0.
        key_img_ = key_img_*255
        key_img_ = key_img_.detach().cpu().numpy()

        cv2.imwrite(os.path.join(mask_folder, f"{d['img_id']:06d}_{obj_name}.png"),key_img_)
        origin_image = d['rgb_crop'][:,:,:3]
        cv2.imwrite(os.path.join(rgb_folder, f"{d['img_id']:06d}_{obj_name}.png"),origin_image)
        cv2.imwrite(os.path.join(visible_mask_folder, f"{d['img_id']:06d}_{obj_name}.png"),mask_)

    def main(self):
        for i, d in enumerate(tqdm(self.data, desc='save embimg', smoothing=0)):
            self.infer(i,d)


sec = save_emb_crop()
sec.main()




