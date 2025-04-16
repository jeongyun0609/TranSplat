import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from tqdm import tqdm
import torch
from glob import glob
from pathlib import Path
import json
from random import random, choice
import imgaug.augmenters as iaa


imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

class CustomDataset(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 mode = 'Training',
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        print("loading data from the directory :",data_root)
        self.data_folder = Path(data_root +'/'+ ('train' if mode=='Training' else 'val'))
        self.name_list = []
        self.label_list = []
        self.visible_mask_list = []
        class_label_json_path = os.path.join(data_root, "class_name.json")
        
        with open(class_label_json_path) as f:
            self.class_label = json.load(f)

        scene_ids = sorted([(p.name) for p in self.data_folder.glob('*')])
        if mode=="Training":
            for scene_id in tqdm(scene_ids, 'loading'):
                scene_folder = self.data_folder / f'{scene_id:s}'
                images = sorted(glob(os.path.join(scene_folder, "rgb_sub/*.png")))
                masks = sorted(glob(os.path.join(scene_folder, "mask_sub/*.png")))
                visible_masks = sorted(glob(os.path.join(scene_folder, "visible_sub/*.png")))

                filter_images = [x for x in images if "N" not in x] 
                filter_masks = [x for x in masks if "N" not in x]
                filter_visible_masks = [x for x in visible_masks if "N" not in x]

                self.name_list += filter_images
                self.label_list += filter_masks
                self.visible_mask_list += filter_visible_masks
                # print(scene_id, len(self.visible_mask_list), len(self.name_list))

 
        else:
            for scene_id in tqdm(scene_ids, 'loading'):
                scene_folder = self.data_folder / f'{scene_id:s}'
                images = sorted(glob(os.path.join(scene_folder, "rgb_sub/*.png")))
                masks = sorted(glob(os.path.join(scene_folder, "mask_sub/*.png")))
                visible_masks = sorted(glob(os.path.join(scene_folder, "visible_sub/*.png")))

                filter_images = [x for x in images if "N" not in x] 
                filter_masks = [x for x in masks if "N" not in x]
                filter_visible_masks = [x for x in visible_masks if "N" not in x]

                self.name_list += filter_images
                self.label_list += filter_masks
                self.visible_mask_list += filter_visible_masks
                # print(scene_id, len(self.visible_mask_list), len(self.name_list))


        if mode=='Training':
            self.data_folder = Path(data_root +'/'+ 'train_pbr1')
            scene_ids = sorted([(p.name) for p in self.data_folder.glob('*')])
            for scene_id in tqdm(scene_ids, 'loading'):
                scene_folder = self.data_folder / f'{scene_id:s}'
                images = sorted(glob(os.path.join(scene_folder, "rgb_sub/*.png")))
                masks = sorted(glob(os.path.join(scene_folder, "mask_sub/*.png")))
                visible_masks = sorted(glob(os.path.join(scene_folder, "visible_sub/*.png")))
                filter_images = [x for x in images if "N" not in x] 
                filter_masks = [x for x in masks if "N" not in x]
                filter_visible_masks = [x for x in visible_masks if "N" not in x]
                self.name_list += filter_images
                self.label_list += filter_masks
                self.visible_mask_list += filter_visible_masks


            self.data_folder = Path(data_root +'/'+ 'train_pbr2')
            scene_ids = sorted([(p.name) for p in self.data_folder.glob('*')])
            for scene_id in tqdm(scene_ids, 'loading'):
                scene_folder = self.data_folder / f'{scene_id:s}'
                images = sorted(glob(os.path.join(scene_folder, "rgb_sub/*.png")))
                masks = sorted(glob(os.path.join(scene_folder, "mask_sub/*.png")))
                visible_masks = sorted(glob(os.path.join(scene_folder, "visible_sub/*.png")))
                filter_images = [x for x in images if "N" not in x] 
                filter_masks = [x for x in masks if "N" not in x]
                filter_visible_masks = [x for x in visible_masks if "N" not in x]

                self.name_list += filter_images
                self.label_list += filter_masks
                self.visible_mask_list += filter_visible_masks

            self.data_folder = Path(data_root +'/'+ 'train_pbr3')
            scene_ids = sorted([(p.name) for p in self.data_folder.glob('*')])
            for scene_id in tqdm(scene_ids, 'loading'):
                scene_folder = self.data_folder / f'{scene_id:s}'
                images = sorted(glob(os.path.join(scene_folder, "rgb_sub/*.png")))
                masks = sorted(glob(os.path.join(scene_folder, "mask_sub/*.png")))
                visible_masks = sorted(glob(os.path.join(scene_folder, "visible_sub/*.png")))
                filter_images = [x for x in images if "N" not in x] 
                filter_masks = [x for x in masks if "N" not in x]
                filter_visible_masks = [x for x in visible_masks if "N" not in x]

                self.name_list += filter_images
                self.label_list += filter_masks
                self.visible_mask_list += filter_visible_masks

        self.mode = mode
        
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.data_root = data_root
        

        self._length = len(self.name_list)
        self.labels = {
            "relative_file_path_": [l for l in self.name_list],
            "file_path_": [l for l in self.name_list],
            "mask_file_path_": [l for l in self.label_list],
            "visible_mask_file_path_": [l for l in self.visible_mask_list],
        }
        self.size = size
        self.interpolation = {
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        
        self.seq = iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 5), size=0.3, squared=False)),         
                            iaa.Sometimes(0.5, iaa.CoarseDropout((0.1, 0.5), size_percent=(0.01, 0.05)))
                            ], random_order=False)   



    def __len__(self):
        return self._length

    def image_and_mask_batch(self, image, mask):
        H,W,C = image.shape
        image_mask = np.zeros((2,H,W,C))
        image_mask[0] = image
        image_mask[1] = mask
        return image_mask
        

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        mask = cv2.imread(example["mask_file_path_"], -1)
        mask_ = np.array(mask).astype(np.uint8)
        crop = min(mask_.shape[0], mask_.shape[1])
        h, w = mask_.shape[0], mask_.shape[1]
        mask_ = mask_[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        mask = Image.fromarray(mask_)
        if self.size is not None:
            mask = mask.resize((self.size, self.size), resample=self.interpolation)

        visible_mask = cv2.imread(example["visible_mask_file_path_"], -1)
        visible_mask_ = np.array(visible_mask).astype(np.uint8)
        crop = min(visible_mask_.shape[0], visible_mask_.shape[1])
        h, w = visible_mask_.shape[0], visible_mask_.shape[1]
        visible_mask_ = visible_mask_[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        visible_mask = Image.fromarray(visible_mask_)
        if self.size is not None:
            visible_mask = visible_mask.resize((self.size, self.size), resample=self.interpolation)


        category = example["file_path_"].split("/")[-1][7:15]
        class_text = self.class_label[category]
        crop_ratio = np.random.rand(1)[0]
        if crop_ratio > 0.7:
            crop_ratio = 0.7    
        crop_axis = np.random.randint(0, 2)
        crop_direction = np.random.randint(0, 2)
        top = 0
        left = 0
        if crop_axis == 0:
            if crop_direction == 0:
                w = w - w * crop_ratio          
            if crop_direction == 1:
                left = w * crop_ratio
                w = w - w * crop_ratio   
        if crop_axis == 1:
            if crop_direction == 0:
                h = h - h * crop_ratio          
            if crop_direction == 1:
                top = h * crop_ratio
                h = h - h * crop_ratio
        right = w + left
        bottom = h + top
        ltrb = np.array([left, top, right, bottom], dtype=np.uint32)
        left, top, right, bottom =ltrb
            
        if torch.rand(1)>0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            visible_mask = transforms.functional.hflip(visible_mask)

        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        visible_mask = np.array(visible_mask).astype(np.uint8)
        visible_mask = np.expand_dims(visible_mask, axis=-1)
        if torch.rand(1)>0.5:
            zero_image = np.zeros(image.shape, dtype=np.uint8)
            zero_mask = np.zeros(mask.shape, dtype=np.uint8)
            zero_visible_mask = np.zeros(visible_mask.shape, dtype=np.uint8)
            crop_image = image[top:bottom,left:right]
            crop_mask = mask[top:bottom,left:right]
            crop_visible_mask = visible_mask[top:bottom,left:right]
            zero_image[int(top):int(bottom),int(left):int(right)] = crop_image
            zero_mask[int(top):int(bottom),int(left):int(right)] = crop_mask
            zero_visible_mask[int(top):int(bottom),int(left):int(right)] = crop_visible_mask
            image = zero_image
            mask = zero_mask
            visible_mask = zero_visible_mask
        image  = image[None]
        if torch.rand(1)>0.5:
            image = self.seq(images=image)
        image = image[0]

        example["mask_image"] = (image / 255).astype(np.float32)
        example["image"] = (mask / 127.5 - 1.0).astype(np.float32) 
        visible_mask[visible_mask>127.5]=255.
        visible_mask[visible_mask<127.5]=0.
        example["mask"] = (visible_mask/255).astype(np.float32)
        example["txt"] = choice(imagenet_templates_small).format(class_text)
        
        ######mask latent######
        example["hint"] = (image / 255)*(visible_mask>127.5).astype(np.float32)
        return example


class TRansPosetrain(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="../stage_0/SurfEmb/data/bop/TRansPose_surfemb", mode = "Training", **kwargs)

class TRansPosetest(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="../stage_0/SurfEmb/data/bop/TRansPose_surfemb", mode = "test", **kwargs)
