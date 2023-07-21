import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision

from p4_helper import *
from utils import reset_seed
from utils.grad import rel_error
from torch.utils.data import DataLoader

import torchvision.models as models
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
from pose_cnn import PoseCNN, FeatureExtraction, SegmentationBranch, TranslationBranch, RotationBranch
feature_extractor = FeatureExtraction(pretrained_model=vgg16)
segmentation_branch = SegmentationBranch()
translation_branch = TranslationBranch()
rotation_branch = RotationBranch()

# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

import multiprocessing

# Set a few constants related to data loading.
NUM_CLASSES = 10
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count()
path = os.getcwd()
PATH = os.path.join(path)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

from utils import PROPSPoseDataset
import utils

utils.reset_seed(0)

def get_data():
  # NOTE: Set `download=True` for the first time when you set up Google Drive folder.
  # Turn it back to `False` later for faster execution in the future.
  # If this hangs, download and place data in your drive manually.
  train_dataset = PROPSPoseDataset(
      PATH, "train",
      download=True #False
  ) 
  val_dataset = PROPSPoseDataset(PATH, "val")
  return train_dataset, val_dataset

def main():
  train_dataset, val_dataset = get_data()
  rob599.reset_seed(0)

  dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
  
  vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
  posecnn_model = PoseCNN(pretrained_backbone = vgg16, 
                  models_pcd = torch.tensor(val_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                  cam_intrinsic = val_dataset.cam_intrinsic).to(DEVICE)
  posecnn_model.load_state_dict(torch.load(os.path.join(PATH, "posecnn_model.pth")))
  num_samples = 5
  for i in range(num_samples):
      out = eval(posecnn_model, dataloader, DEVICE)
  
      plt.axis('off')
      plt.imshow(out)
      plt.show()

if __name__ == '__main__':
    main()
