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
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        

# %%
supervised='True'
problem = "Tomography"
problem_desc = 'Denoising_on_CT'
sigma = .1
save_dir = f'./datasets/{problem}'
path = save_dir
imwidth = 256
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
train = True  

if not os.path.exists(f'{save_dir}/dinv_dataset0.h5'):  # download dataset
    save_dir2 = './datasets/'
    hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/dinv_dataset0.h5',
                    cache_dir=save_dir2, local_dir=save_dir2)
    # hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/physics0.pt',
    #                 cache_dir=save_dir2, local_dir=save_dir2)

# defined physics
physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma))

# load dataset

data_train = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=True)]
data_test = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)]

if supervised:

        
    backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=4, bias=False, batch_norm=False).to(device)
    # backbone = dinv.models.GSDRUNet(act_mode='s', in_channels=1, out_channels=1, 
                                    # pretrained="download").to(device)
    model = dinv.models.Denoiser(backbone).to(device)



    if not train:
        ckp_path = f'{save_dir}/sup/ckp.pth.tar'
        download_model(f'{problem}/sup/ckp.pth.tar')
        model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
        model.eval()
        num_params = count_parameters(model)
        print(f"Number of trainable parameters: {num_params}")
        print('Model loaded')


else:
    # choose a reconstruction architecture
    backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=5, # 4
                                bias=False, batch_norm=False).to(device)
    model = dinv.models.Denoiser(backbone, pinv=True).to(device)


    if not train:
        ckp_path = f'{save_dir}/rei/ckp.pth.tar' #
        download_model(f'{problem}/rei/ckp.pth.tar')
        model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
        model.eval()
        num_params = count_parameters(model)
        print(f"Number of trainable parameters: {num_params}")
        print('Model loaded')


batch_size = 8

# %%
import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

wandb_vis = True #True

method = 'sup' #sure'




# number of training epochs
epochs = 200

# set seed
torch.manual_seed(0)

print('length of data_train:', len(data_train))
print('length of data_test:', len(data_test))
print('length of data_train[0]:', len(data_train[0]))
print('length of data_test[0]:', len(data_test[0]))

###
if not isinstance(data_train, list):
    data_train = [data_train]

if not isinstance(data_test, list):
    data_test = [data_test]

losses = [dinv.loss.SupLoss(metric=torch.nn.MSELoss())]
# _, losses = get_losses(method, model, physics, problem, device,NE_model=False) #Choose your suitable loss with respect to your setting
print('method', method)
print('losses', losses)
num_params = count_parameters(model)
print(f"Number(True) of trainable parameters: {num_params}")
print('Model loaded')


train_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) for data in data_train]
test_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False) for data in data_test]

intial_lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=intial_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.5), gamma=0.5)

if wandb_vis:
        wandb.init(
            # set the wandb project where this run will be logged
            project="denoising_on_CT",
            # track hyperparameters and run metadata
            config={
                "learning_rate": intial_lr,
                "architecture": "UNrolled-UNet",
                "dataset": f'{problem}',
                "name": f'{problem_desc}',
                "epochs": epochs,
                'batch_size': batch_size,
                'method': method,
            }
        )


metrics = [dinv.loss.PSNR()]
trainer = dinv.Trainer(losses=losses, model=model, ckp_interval=1, online_measurements=False,
                       physics=physics, verbose_individual_losses=True, metrics=metrics,
                       save_path=path+f'{method}/', wandb_vis=wandb_vis, plot_images=True, freq_plot=int(epochs/20),
                       scheduler=scheduler, optimizer=optimizer, train_dataloader=train_dataloader,
                       device=device, eval_dataloader=test_dataloader, eval_interval=int(epochs/20), epochs=epochs)


trainer.train()



