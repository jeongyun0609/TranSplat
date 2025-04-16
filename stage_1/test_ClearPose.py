import argparse, os, sys, glob, datetime, glob, importlib, csv
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from cldm.ddim_hacked import DDIMSampler


import numpy as np
import time
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset, IterableDataset
from functools import partial
from PIL import Image
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger
from minlora import add_lora, LoRAParametrization
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, LoraCheckpoint
import einops
from abc import abstractmethod
import cv2
import json

def pad_image(image, x, y):
    h, w = image.shape[:2]
    top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0

    if w > h:
        if y == 0:
            top_pad = w-h
        else:
            bottom_pad = w-h
    else:
        if x==0:
            left_pad = h-w
        else:
            right_pad = h-w
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image, top_pad, bottom_pad, left_pad, right_pad

def make_batch_sd(
        image,
        mask_visib,
        origin_image,
        txt,
        bbox,
        device):    
    origin_images = []
    mask_visibs = []
    masked_images = []
    object_num = 0
    padding_list = []
    image_wh = []
    for i in range(len(txt)):
        pre_crop_image = cv2.imread(origin_image[i], -1)
        origin_image_ = cv2.imread(origin_image[i], -1)
        top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0
        [x,y,width,height] = bbox[i]
        if width <10  or height <10:
            continue
        object_num+=1
        mask_visib_ = cv2.imread(mask_visib[i], 0)
        mask_visib_ = mask_visib_[y:y+height, x:x+width]
        mask_visib__ = mask_visib_
        mask_visib__[mask_visib__>0] = 1.
        mask_visib__[mask_visib__==0] = 0.
        mask_visib_[mask_visib_>0] = 1.
        mask_visib_[mask_visib_==0] = 0.
        crop_image = np.zeros((height, width,3))
        crop_image = pre_crop_image[y:y+height, x:x+width]           
        if width/height>1.1 or height/width>1.1:
            crop_image, top_pad, bottom_pad, left_pad, right_pad = pad_image(crop_image, x, y)
            mask_visib_, top_pad, bottom_pad, left_pad, right_pad = pad_image(mask_visib_,  x, y)
            height = crop_image.shape[1]
            width = crop_image.shape[0]
        image__ = cv2.resize(crop_image, (256,256))
        image = np.array(image__)
        image = (image / 255.0).astype(np.float32)
        image = torch.from_numpy(image).to(dtype=torch.float32)
        mask_ = cv2.resize(mask_visib_, (256,256))
        mask_ = torch.from_numpy(mask_).to(dtype=torch.float32)
        mask_ = mask_.unsqueeze(dim=-1)
        mask_ = torch.cat([mask_,mask_,mask_], dim=-1)
        masked_image = image*mask_
        mask = cv2.resize(mask_visib_, (32,32))
        mask = torch.from_numpy(mask).to(dtype=torch.float32)
        mask = mask.unsqueeze(dim=0)
        mask_visibs.append(mask_visib__)
        # cv2.imwrite(f"{i}.png",image__)
        if object_num==1:
            images = image.unsqueeze(dim=0)
            masks = mask.unsqueeze(dim=0)
            masked_images = masked_image.unsqueeze(dim=0)
        else:
            image = image.unsqueeze(dim=0)
            images = torch.cat([images, image], dim=0)
            mask = mask.unsqueeze(dim=0)
            masks = torch.cat([masks, mask], dim=0)
            masked_image = masked_image.unsqueeze(dim=0)
            masked_images = torch.cat([masked_images, masked_image], dim=0)
        padding_list.append([top_pad, bottom_pad, left_pad, right_pad])
        image_wh.append([width, height])
        origin_images.append(origin_image_)
        txt[i] = txt[i] + ', best quality'
    images = einops.rearrange(images, 'b h w c -> b c h w').clone()
    masked_images = einops.rearrange(masked_images, 'b h w c -> b c h w').clone()
    batch = {
            "jpg": images.to(device=device),
            "txt": txt,
            "origin_image": origin_images,
            "mask_visib":mask_visibs,
            "mask": masks.to(device=device),
            "maksed_image": masked_images.to(device=device),
            "image_wh": image_wh,
            "padding_list": padding_list
            }

    return batch

def make_padding_bb(bb_xywh, render_dimension=[640, 480, 3], pad_factor=1):
    x, y, w, h = np.array(bb_xywh).astype(np.int32)                               
    size = int(np.maximum(h, w) * pad_factor)                
    if x+w/2-size/2 < 0:
        left = 0
        right = np.minimum(x+w/2+size/2, render_dimension[0])
    elif x+w/2+size/2 > render_dimension[0]:
        right = render_dimension[0]
        left = np.maximum(x+w/2-size/2, 0)
    else:
        left = np.maximum(x+w/2-size/2, 0)
        right = np.minimum(x+w/2+size/2, render_dimension[0])
        
    if y+h/2-size/2 < 0:
        top = 0
        bottom = np.minimum(y+h/2+size/2, render_dimension[1])
    elif y+h/2+size/2 > render_dimension[1]:
        bottom = render_dimension[1]
        top = np.maximum(y+h/2-size/2, 0)
    else:
        top = np.maximum(y+h/2-size/2, 0)
        bottom = np.minimum(y+h/2+size/2, render_dimension[1])            

    return np.array([int(left), int(top), int(right)-int(left), int(bottom)-int(top)]) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ClearPose_syn",
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta of ddim",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=6.0,
        help="scale of unconditional guidance",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="SurfEmb_sd.ckpt",
        help="scale of unconditional guidance",
    )
    opt = parser.parse_args()
    indir = os.path.join("../stage_0/SurfEmb/data/bop/", opt.dataset, f"syn_test_{opt.index:02d}")
    outdir = os.path.join("../stage_0/SurfEmb/data/bop/", opt.dataset, f"syn_test_{opt.index:02d}/SurfEmb")
    ckpt=opt.ckpt
    images = sorted(glob.glob(os.path.join(indir,"rgb/*.png")))
    model = create_model(f'./models/SurfEmb.yaml').cpu()
    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=64)
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=64)
        },
        torch.nn.Conv2d: {
            "weight": partial(LoRAParametrization.from_conv2d, rank=64)
        }
    }
    for name, module in model.model.diffusion_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)
    for name, module in model.control_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)
    add_lora(model.cond_stage_model, lora_config=lora_config)
    model.load_state_dict(load_state_dict(f'./models/{ckpt}', location='cpu'), strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.cuda()  
    sampler = DDIMSampler(model)
    indices = [i for i in range(len(images))]
    images = [images[i] for i in indices]

    scene_gt_json_path = os.path.join(indir, "scene_gt.json")
    scene_gt_info_json_path = os.path.join(indir, "scene_gt_info.json")
    class_label_json_path = os.path.join("../stage_0/SurfEmb/data/bop",opt.dataset,"class_label.json")

    obj_key_path = os.path.join("/mydata/data/JYK/clearpose/models/obj_key.json")
    with open(obj_key_path) as f:
        obj_key = json.load(f)
    with open(scene_gt_json_path) as f:
        scene_gt_json = json.load(f)
    with open(scene_gt_info_json_path) as f:
        scene_gt_info_json = json.load(f)
    with open(class_label_json_path) as f:
        class_label_json = json.load(f)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "remove_bg"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "SurfEmb"), exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            for index, image in enumerate(tqdm(images, total=len(images))):
                if index%opt.batch!=0:
                    continue
                reindex = index+1
                text_list = []
                bbox_list = []
                mask_visib_image_path_list = []
                n_prompt_list = []
                n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality'
                outpath_ = []
                outpath___ = []
                origin_image_path = []
                object_num_= []
                image_ = []
                for j in range(opt.batch):
                    image_index = reindex + j
                    l=0
                    object_num = len(scene_gt_info_json[str(image_index)])
                    bbox_info = scene_gt_info_json[str(image_index)]
                    obj_id_info = scene_gt_json[str(image_index)]
                    for k in range(object_num):
                        mask_visib_image_path = os.path.join(indir,"mask", f"{image_index:06d}_{k:06d}"+".png")
                        object_category = obj_id_info[k]["obj_id"]
                        try:
                            mask_visible = cv2.imread(mask_visib_image_path, 0 )
                            nonzero = np.where(mask_visible>0)
                        except TypeError:
                            continue
                        try:
                            y_min, x_min = np.min(nonzero, axis=1)
                            y_max, x_max = np.max(nonzero, axis=1)
                            width = x_max - x_min
                            height = y_max - y_min
                            x = x_min
                            y = y_min
                            [x,y,width,height] = make_padding_bb([x,y,width,height])
                        except ValueError:
                            continue
                        if width <10  or height <10:
                            continue
                        text = class_label_json[str(obj_key[object_category])]
                        text_list.append(text)
                        bbox_list.append([x,y,width,height])
                        mask_visib_image_path_list.append(mask_visib_image_path)
                        n_prompt_list.append(n_prompt)
                        origin_image_path.append(os.path.join(indir,"rgb",f"{image_index:06d}"+".png") )
                        l+=1
                    outpath_.append(os.path.join(outdir, "remove_bg",f"{image_index:06d}"+".png"))
                    outpath___.append(os.path.join(outdir, "SurfEmb",f"{image_index:06d}"+".png"))
                    object_num_.append(l)

                batch = make_batch_sd(image, mask_visib = mask_visib_image_path_list, origin_image = origin_image_path, txt=text_list, bbox = bbox_list, device=device)
                c = model.cond_stage_model.encode(batch["txt"])
                num = len(batch["txt"])
                control = batch["jpg"]
                masks = batch['mask']
                masked_images = batch['maksed_image']
                a_prompt = 'best quality'
                masked_latent = model.get_first_stage_encoding(model.encode_first_stage(masked_images))
                cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(batch["txt"])], "mask": [masks], "masked_latent": [masked_latent]}
                un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(n_prompt_list)], "mask": [masks], "masked_latent": [masked_latent]}
                b, channels, h, w = cond["c_concat"][0].shape
                shape = (model.channels,h // 8, w // 8)
                uc_cross = model.get_unconditional_conditioning(num)
                uc_full = {"c_concat": [control], "c_crossattn": [uc_cross], "mask": [masks], "masked_latent": [masked_latent]}
                model.control_scales = [1.0] * 13              
                samples_cfg, intermediates = sampler.sample(
                                    opt.steps,
                                    num,
                                    shape,
                                    cond,
                                    verbose=False,
                                    eta=opt.eta,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc_full,
                            )
                x_samples_ddim = model.decode_first_stage(samples_cfg)
                predicted_image = (einops.rearrange(x_samples_ddim, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                index = 0
                image_num = len(outpath_)
                for k in range(image_num):
                    full_only_object_ = np.zeros((480,640,4))
                    full_only_object = np.zeros((480,640,4))
                    full_color = np.zeros((480,640,3))
                    for j in range(object_num_[k]):
                        [x,y,w,h] = bbox_list[index]
                        [w_,h_] = batch["image_wh"][index]
                        [top_pad, bottom_pad, left_pad, right_pad] = batch["padding_list"][index]
                        if w <10  or h <10:
                            continue
                        predicted_image_ = Image.fromarray(predicted_image[index])
                        predicted_image_ = predicted_image_.resize((w_,h_))
                        predicted_image_np = np.array(predicted_image_)
                        predicted_image_np = predicted_image_np[top_pad:h_-bottom_pad,left_pad:w_-right_pad]
                        h, w, c = predicted_image_np.shape
                        colorize_image = batch["origin_image"][index]
                        H,W,C = colorize_image.shape
                        mask = np.stack((batch["mask_visib"][index]>0, batch["mask_visib"][index]>0,batch["mask_visib"][index]>0), axis = -1)
                        MASK = np.sum(predicted_image_np>50, axis=-1)
                        mask__ = np.stack((MASK!=0, MASK!=0,MASK!=0), axis = -1)
                        only_object = np.zeros((H,W,4))
                        only_object[y:y+h, x:x+w,:3] =predicted_image_np*mask*mask__
                        only_object[y:y+h, x:x+w,3] = mask[:,:,0]*255*mask__[:,:,0]
                        only_object_ = np.zeros(only_object.shape)
                        mask_ = np.stack((only_object[:,:,3]>0, only_object[:,:,3]>0,only_object[:,:,3]>0), axis = -1)
                        only_object_[:,:,:3] = batch["origin_image"][index][:,:,:3]*mask_
                        only_object_[:,:,3] = only_object[:,:,3]
                        full_only_object_ = full_only_object_+only_object_
                        full_only_object = full_only_object+only_object
                        MASK = np.stack([full_only_object_[:,:,3]>0,full_only_object_[:,:,3]>0,full_only_object_[:,:,3]>0], axis = -1)
                        full_color = full_only_object[:,:,:3]*MASK+batch["origin_image"][index]*(1-MASK)
                        index +=1
                    cv2.imwrite(outpath_[k], full_only_object)
                    cv2.imwrite(outpath___[k], full_color)
