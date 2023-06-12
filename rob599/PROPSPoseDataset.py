import os
import json
from typing import Any, Callable, Optional, Tuple
import random

import cv2
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from rob599 import Visualize, chromatic_transform, add_noise



class PROPSPoseDataset(Dataset):
    
    base_folder = "PROPS-Pose-Dataset"
    url = "https://drive.google.com/file/d/15rhwXhzHGKtBcxJAYMWJG7gN7BLLhyAq/view?usp=share_link"
    filename = "PROPS-Pose-Dataset.tar.gz"
    tgz_md5 = "a0c39fe326377dacd1d652f9fe11a7f4"
    
    def __init__(
        self, 
        root: str,
        split: str = 'train', 
        download: bool = False,
        ) -> None:
        assert split in ['train', 'val']
        
        self.root = root
        self.split = split
        self.dataset_dir = os.path.join(self.root, self.base_folder)
        
        if download:
            self.download()
            
            

        ## parameter
        self.max_instance_num = 10
        self.H = 480
        self.W = 640
        self.rgb_aug_prob = 0.4
        self.cam_intrinsic = np.array([
                            [902.19, 0.0, 342.35],
                            [0.0, 902.39, 252.23],
                            [0.0, 0.0, 1.0]])
        self.resolution = [640, 480]

        self.all_lst = self.parse_dir()
        self.shuffle()
        self.models_pcd = self.parse_model()


        self.obj_id_list = [
            1, # master chef
            2, # cracker box
            3, # sugar box
            4, # soup can
            5, # mustard bottle
            6, # tuna can
            8, # jello box
            9, # meat can
            14,# mug
            18 # marker
        ]
        self.id2label = {}
        for idx, id in enumerate(self.obj_id_list):
            self.id2label[id] = idx + 1

    def parse_dir(self):
      data_dir = os.path.join(self.dataset_dir, self.split)
      rgb_path = os.path.join(data_dir, "rgb")
      depth_path = os.path.join(data_dir, "depth")
      mask_path = os.path.join(data_dir, "mask_visib")
      scene_gt_json = os.path.join(data_dir, self.split+"_gt.json")
      scene_gt_info_json = os.path.join(data_dir, self.split+"_gt_info.json")
      rgb_list = os.listdir(rgb_path)
      rgb_list.sort()
      depth_list = os.listdir(depth_path)
      depth_list.sort()
      mask_list = os.listdir(mask_path)
      mask_list.sort()
      scene_gt = json.load(open(scene_gt_json))
      scene_gt_info = json.load(open(scene_gt_info_json))
      assert len(rgb_list) == len(depth_list) == len(scene_gt) == len(scene_gt_info), "data files number mismatching"
      all_lst = []
      for rgb_file in rgb_list:
          idx = int(rgb_file.split(".png")[0])
          depth_file = f"{idx:06d}.png"
          scene_objs_gt = scene_gt[str(idx)]
          scene_objs_info_gt = scene_gt_info[str(idx)]
          objs_dict = {}
          for obj_idx in range(len(scene_objs_gt)):
              objs_dict[obj_idx] = {}
              objs_dict[obj_idx]['R'] = np.array(scene_objs_gt[obj_idx]['cam_R_m2c']).reshape(3, 3)
              objs_dict[obj_idx]['T'] = np.array(scene_objs_gt[obj_idx]['cam_t_m2c']).reshape(3, 1)
              objs_dict[obj_idx]['obj_id'] = scene_objs_gt[obj_idx]['obj_id']
              objs_dict[obj_idx]['bbox_visib'] = scene_objs_info_gt[obj_idx]['bbox_visib']
              assert f"{idx:006d}_{obj_idx:06d}.png" in mask_list
              objs_dict[obj_idx]['visible_mask_path'] = os.path.join(mask_path, f"{idx:006d}_{obj_idx:06d}.png")
          """
          obj_sample = (rgb_path, depth_path, objs_dict)
          objs_dict = {
              0: {
                  cam_R_m2c:
                  cam_t_m2c:
                  obj_id:
                  bbox_visib:
                  visiable_mask_path:
              }
              ...
          }
          """
          obj_sample = (
              os.path.join(rgb_path, rgb_file),
              os.path.join(depth_path, depth_file),
              objs_dict
          )
          all_lst.append(obj_sample)
      return all_lst

    def parse_model(self):
        model_path = os.path.join(self.dataset_dir, "model")
        objpathdict = {
            1: ["master_chef_can", os.path.join(model_path, "1_master_chef_can", "textured_simple.obj")],
            2: ["cracker_box", os.path.join(model_path, "2_cracker_box", "textured_simple.obj")],
            3: ["sugar_box", os.path.join(model_path, "3_sugar_box", "textured_simple.obj")],
            4: ["tomato_soup_can", os.path.join(model_path, "4_tomato_soup_can", "textured_simple.obj")],
            5: ["mustard_bottle", os.path.join(model_path, "5_mustard_bottle", "textured_simple.obj")],
            6: ["tuna_fish_can", os.path.join(model_path, "6_tuna_fish_can", "textured_simple.obj")],
            7: ["gelatin_box", os.path.join(model_path, "8_gelatin_box", "textured_simple.obj")],
            8: ["potted_meat_can", os.path.join(model_path, "9_potted_meat_can", "textured_simple.obj")],
            9: ["mug", os.path.join(model_path, "14_mug", "textured_simple.obj")],
            10: ["large_marker", os.path.join(model_path, "18_large_marker", "textured_simple.obj")],
        }
        self.visualizer = Visualize(objpathdict, self.cam_intrinsic, self.resolution)
        models_pcd_dict = {index:np.array(self.visualizer.objnode[index]['mesh'].vertices) for index in self.visualizer.objnode}
        models_pcd = np.zeros((len(models_pcd_dict), 1024, 3))
        for m in models_pcd_dict:
            model = models_pcd_dict[m]
            models_pcd[m - 1] = model[np.random.randint(0, model.shape[0], 1024)]
        return models_pcd

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        """
        obj_sample = (rgb_path, depth_path, objs_dict)
        objs_dict = {
            0: {
                cam_R_m2c:
                cam_t_m2c:
                obj_id:
                bbox_visib:
                visiable_mask_path:
            }
            ...
        }

        data_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs',
            'centermaps', []
        }
        """
        rgb_path, depth_path, objs_dict = self.all_lst[idx]
        data_dict = {}
        with Image.open(rgb_path) as im:
            rgb = np.array(im)

        if self.split == 'train' and np.random.rand(1) > 1 - self.rgb_aug_prob:
            rgb = chromatic_transform(rgb)
            rgb = add_noise(rgb)
        rgb = rgb.astype(np.float32)/255
        data_dict['rgb'] = rgb.transpose((2,0,1))

        with Image.open(depth_path) as im:
            data_dict['depth'] = np.array(im)[np.newaxis, :]
        ## TODO data-augmentation of depth 
        assert(len(objs_dict) <= self.max_instance_num)
        objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))
        centermaps = np.zeros((self.max_instance_num, 3, self.resolution[1], self.resolution[0]))
        ## test
        img = cv2.imread(rgb_path)

        for idx in objs_dict.keys():
            if len(objs_dict[idx]['bbox_visib']) > 0:
                ## have visible mask 
                objs_id[idx] = self.id2label[objs_dict[idx]['obj_id']]
                assert(objs_id[idx] > 0)
                with Image.open(objs_dict[idx]['visible_mask_path']) as im:
                    label[objs_id[idx]] = np.array(im, dtype=bool)
                ## [x_min, y_min, width, height]
                bbx[idx] = objs_dict[idx]['bbox_visib']
                RT = np.zeros((4, 4))
                RT[3, 3] = 1
                RT[:3, :3] = objs_dict[idx]['R']
                RT[:3, [3]] = objs_dict[idx]['T']
                RT = np.linalg.inv(RT)                
                RTs[idx] = RT[:3]
                center_homo = self.cam_intrinsic @ RT[:3, [3]]
                center = center_homo[:2]/center_homo[2]
                x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
                y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
                xv, yv = np.meshgrid(x, y)
                dx, dy = center[0] - xv, center[1] - yv
                distance = np.sqrt(dx ** 2 + dy ** 2)
                nx, ny = dx / distance, dy / distance
                Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]
                centermaps[idx] = np.array([nx, ny, Tz])
                ## test
                img = cv2.circle(img, (int(center[0]), int(center[1])), radius=2, color=(0, 0, 255), thickness = -1)
                centers[idx] = np.array([int(center[0]), int(center[1])])
        label[0] = 1 - label[1:].sum(axis=0)
        # Image.fromarray(label[0].astype(np.uint8) * 255).save("testlabel.png")
        # Image.open(rgb_path).save("testrgb.png")
        # cv2.imwrite("testcenter.png", img)
        data_dict['objs_id'] = objs_id
        data_dict['label'] = label
        data_dict['bbx'] = bbx
        data_dict['RTs'] = RTs
        data_dict['centermaps'] = centermaps.reshape(-1, self.resolution[1], self.resolution[0])
        data_dict['centers'] = centers
        return data_dict

    def shuffle(self):
        random.shuffle(self.all_lst)

    def download(self) -> None:
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
