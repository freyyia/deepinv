# %%
import torchvision
from torchvision import datasets, transforms
import deepinv as dinv
from deepinv.utils.demo import load_dataset, load_degradation
from pathlib import Path
import os
from PIL import Image
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import random
import numpy as np
from torchvision.transforms.functional import rotate
from tqdm import tqdm
import torch
import torch.nn.functional as F

from deepinv.utils import get_data_home, load_degradation
from deepinv.models.utils import get_weights_url

# %%
def download_model(path):
    if not os.path.exists(path):
        save_dir2 = './datasets/'
        hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=path,
                        cache_dir=save_dir2, local_dir=save_dir2)
        


# %%
supervised='True'
problem = "Tomography"
problem_desc = 'Denoising_on_CT'
sigma = .1
save_dir = f'./datasets/{problem}'
imwidth = 256
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
train = True  

if not os.path.exists(f'{save_dir}/dinv_dataset0.h5'):  # download dataset
    save_dir2 = './datasets/'
    hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/dinv_dataset0.h5',
                    cache_dir=save_dir2, local_dir=save_dir2)
    # hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/physics0.pt',
    #                 cache_dir=save_dir2, local_dir=save_dir2)



# load dataset
data_train = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=True)]
data_test = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)]

# get iterator
loader = iter(data_train[0])

dummy = 10
save_dir = "/users/tk2017/code/bregman_sampling/Prox_GSPnP/GS_denoising/datasets/LIDC/"
counter = 0

# iterate until exhausted
for x, _ in loader:
    # do something with x
    #print(x.shape)
    # dummy -= 1
    # if dummy == 0:
    #     break

    # x shape: (1, H, W) -> squeeze channel dimension
    img = x.squeeze(0)  # (H, W)

    # scale [0,1] -> [0,255] and convert to uint8
    img = (img * 255).clamp(0, 255).byte()

    # convert to numpy -> PIL
    pil_img = Image.fromarray(img.cpu().numpy())

    # save as PNG
    pil_img.save(os.path.join(save_dir, f"image_{counter:05d}.png"))
    counter += 1

print("finished")
#dinv.utils.plot(x, save_dir=".datasets/tomo")

#%%
# get iterator
loader = iter(data_test[0])

dummy = 10
save_dir = "/users/tk2017/code/bregman_sampling/Prox_GSPnP/GS_denoising/datasets/LIDC_test/"
counter = 0

# iterate until exhausted
for x, _ in loader:
    # do something with x
    #print(x.shape)
    # dummy -= 1
    # if dummy == 0:
    #     break

    # x shape: (1, H, W) -> squeeze channel dimension
    img = x.squeeze(0)  # (H, W)

    # scale [0,1] -> [0,255] and convert to uint8
    img = (img * 255).clamp(0, 255).byte()

    # convert to numpy -> PIL
    pil_img = Image.fromarray(img.cpu().numpy())

    # save as PNG
    pil_img.save(os.path.join(save_dir, f"image_{counter:05d}.png"))
    counter += 1

print("finished")
# %%
