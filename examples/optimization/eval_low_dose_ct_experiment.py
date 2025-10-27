import deepinv as dinv
from pathlib import Path
import torch
import ast
from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils.plotting import plot, plot_curves, plot_inset
from huggingface_hub import hf_hub_download
from deepinv.utils.parameters import get_GSPnP_params
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import wandb
import sys 
import argparse
import matplotlib.pyplot as plt

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def hu_normalize_resize(img: torch.Tensor, hu_min=-1024, hu_max=400, eps=1e-4, size=256):
    """
    Normalize CT image in Hounsfield units to [1e-4, 1.0].
    
    img: torch.Tensor of HU values
    hu_min, hu_max: window for clipping (default: lung window)
    eps: lowest value after scaling
    """
    if img.ndim == 2:
        img = img.unsqueeze(0)  # add channel if missing

    # clip to window
    img = torch.clamp(img, hu_min, hu_max)
    
    # scale to [0,1]
    img = (img - hu_min) / (hu_max - hu_min)
    
    # rescale to [eps, 1.0]
    img = img * (1.0 - eps) + eps

    resize = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)
    img = resize(img)
    img.clip(eps, 1.0)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="My experiment runner")

    # add your arguments here
    # im 0: out of distribution image (CT from LITS)
    # im 1: in-distribution image (CT from LIDC) 0
    # im 2: in-distribution image (CT from LIDC) 3
    # im 3: in-distribution image (CT from LIDC) 40
    parser.add_argument("--im_idx", type=int, default=1)
    parser.add_argument("--prior_type", type=int, default=0)
    parser.add_argument("--method", type=str, default="SKROCK")
    parser.add_argument("--poisson_level", type=float, default=10.0)
    parser.add_argument("--regularization", type=float, default=4.0)
    parser.add_argument("--step_size", type=float, default=5e-5)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--inset_loc", type=ast.literal_eval, default=(0.52, 0.55))

    args, unknown = parser.parse_known_args(sys.argv[1:])

    return args




args = parse_args()
prior_types = ["gspnp", "proxgspnp", "unet"]
BASE_DIR = Path(".")
img_size = 256

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


if args.im_idx == 0:
    #Read sample
    sample = 63
    sample_ct   = read_nii("datasets/lits-test-subset/test-volume-" + str(sample) + ".nii")

    temp = sample_ct[...,70]
    temp_t = torch.from_numpy(np.ascontiguousarray(temp)).float()

    x = (hu_normalize_resize(temp_t)).unsqueeze(0).clamp(1e-4, None).to(device)
else: 
    #lidc test data set
    problem = "Tomography"
    save_dir = f'datasets/{problem}'

    data_test = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)]
    iterator = iter(data_test[0])

    if args.im_idx == 1:
        end_idx = 1
    elif args.im_idx == 2:
        end_idx = 3
    elif args.im_idx == 3:
        end_idx = 40

    for i in range(end_idx):
        (x, _) = next(iterator)

    x = x.unsqueeze(0).to(device)


# Generate a physics operator for CT
num_angles = 360
noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0/args.poisson_level))
physics = dinv.physics.Tomography(angles=num_angles,
    img_width=img_size,
    device=device, 
    noise_model=noise_model,
)


RESULTS_DIR = BASE_DIR / "results" / (
    f"low_dose_ct_{args.method}_im_{args.im_idx}_{prior_types[args.prior_type]}_pl_{args.poisson_level}_reg_{args.regularization}_step_{args.step_size}_it_{args.iterations}"
)

from PIL import Image

#load mean and std from results dir if available
if RESULTS_DIR.exists():
    print(f"Loading results from {RESULTS_DIR}")
    mean = torch.load(RESULTS_DIR / "mean.pt").squeeze()
    std = torch.load(RESULTS_DIR / "std.pt").squeeze()
    pil_img = Image.open(RESULTS_DIR / "STD" / "0.png")#.convert("RGB")
    std2 = np.array(pil_img) / 255.0        # [H, W, 3] in [0,1]
    std2 = torch.from_numpy(std2).permute(2, 0, 1).float()  # → [3,H,W]


import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



# === helper: sepblockfun equivalent in PyTorch ===
def sepblockfun(tensor: torch.Tensor, block_size: tuple, fun=torch.mean):
    """
    Mimics MATLAB's sepblockfun by applying function over non-overlapping blocks.
    tensor: 2D torch.Tensor
    block_size: (bh, bw)
    """
    h, w = tensor.shape
    bh, bw = block_size
    assert h % bh == 0 and w % bw == 0, "Image size must be divisible by block size"

    reshaped = tensor.view(h // bh, bh, w // bw, bw)
    # move blocks together: (num_blocks_h, num_blocks_w, bh, bw)
    reshaped = reshaped.permute(0, 2, 1, 3)
    # apply mean over block pixels
    out = fun(reshaped, dim=(-1, -2))
    return out

# === global matplotlib settings ===
plt.rcParams.update({
    "font.size": 26
})
import torch

def tensor_from_data(img: torch.Tensor, clim=None):
    """
    Convert raw data to a [1,H,W] tensor (grayscale).
    
    img   : torch.Tensor [H,W]
    clim  : tuple (vmin,vmax) for normalization (optional)
    """
    tensor = img.clone().float()

    if clim is not None:
        vmin, vmax = clim
        tensor = (tensor - vmin) / (vmax - vmin)  # normalize to [0,1]
        tensor = torch.clamp(tensor, 0, 1)

    # add channel dimension for consistency → [1,H,W]
    return tensor.unsqueeze(0)


scales = [1, 2, 4, 8, 16]

for v in scales:
    # block sizes are doubled like in MATLAB
    block_size = (v , v )

    std_ = sepblockfun(std, block_size)


    # === plotting ===
    def plot_and_save(img, clim, fname):
        plt.figure()
        plt.imshow(img.numpy(), cmap="gray", vmin=clim[0], vmax=clim[1])
        plt.axis("off")
        #plt.colorbar()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches="tight", dpi=150)
        plt.show()

    plot_and_save(std_, [std.min(), 0.1], f"uq_vis_{v}.png")
    res = tensor_from_data(std_, clim=[std2.min(), std2.max()])
    plot(res, rescale_mode="clip")



plot(tensor_from_data(std))
plot(std2)


print("range of mean: ", mean.min().item(), mean.max().item())
print("range of std: ", std.min().item(), std.max().item())


# print(f"initial PSNR: {dinv.metric.PSNR()(x, physics.A_dagger(y).clamp(0,1)).item():.2f} dB")
# print(f"reconstruction PSNR: {dinv.metric.PSNR()(x, mean.clamp(0,1)).item():.2f} dB")

# print(f"reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, mean.clamp(0,1)).item():.3f}")


# # plot images. Images are saved in RESULTS_DIR.
# imgs = [y, x, physics.A_dagger(y), mean]
# plot(
#     imgs,
#     titles=["Input", "GT", "FBP", "recon"],
#     save_dir=RESULTS_DIR,
#     rescale_mode="min_max",
#     #rescale_mode="clip"
# )

# plot([torch.sqrt(var)], titles=["STD"], save_dir=RESULTS_DIR)

# mean = torch.load(RESULTS_DIR / "mean.pt")
# std = torch.load(RESULTS_DIR / "std.pt")

# plot([std], titles=["STD_eval"], save_dir=RESULTS_DIR)


# print("range of std: ", std.min().item(), std.max().item())
# import matplotlib.pyplot as plt
# plt.imshow(std.squeeze(), vmax=0.1207, cmap='gray')
# plt.colorbar()

# # %% plot inset
    
# dinv.utils.plot_inset( imgs,
#     titles=["Observation", "GT", "FBP", "Recon"],
#     extract_loc=args.inset_loc,
#     inset_loc=(0.0, 0.6),
#     save_fn = RESULTS_DIR / "inset.png",
#     )
# # %%

# dinv.utils.plot_inset( [y],
#     titles=[""],
#     extract_loc=args.inset_loc,
#     inset_loc=(0.0, 0.6),
#     save_fn = RESULTS_DIR / "obs_inset.png",
#     )

# dinv.utils.plot_inset( [physics.A_dagger(y)],
#     titles=[""],
#     extract_loc=args.inset_loc,
#     inset_loc=(0.0, 0.6),
#     save_fn = RESULTS_DIR / "fbp_inset.png",
#     )

# dinv.utils.plot_inset( [x],
# titles=[""],
# extract_loc=args.inset_loc,
# inset_loc=(0.0, 0.6),
# save_fn = RESULTS_DIR / "gt_inset.png",
# )

# dinv.utils.plot_inset( [mean],
#     titles=[""],
#     extract_loc=args.inset_loc,
#     inset_loc=(0.0, 0.6),
#     save_fn = RESULTS_DIR / "recon_inset.png",
#     )


