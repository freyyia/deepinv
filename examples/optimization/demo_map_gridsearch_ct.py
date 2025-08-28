import deepinv as dinv
from pathlib import Path
import torch

from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils.plotting import plot, plot_curves, plot_inset
from huggingface_hub import hf_hub_download
from deepinv.utils.parameters import get_GSPnP_params
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

# %%
# Helpers for Lung data set.

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


#commenting use of lung data set

# Read sample
# sample = 63
# sample_ct   = read_nii("../../../deepinv/examples/optimization/datasets/lits-test-subset/test-volume-" + str(sample) + ".nii")

# temp = sample_ct[...,70]
# temp_t = torch.from_numpy(np.ascontiguousarray(temp)).float()
# temp_t = (hu_normalize_resize(temp_t)).unsqueeze(0).clamp(1e-4, None)

# plot(temp_t)


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results" / "demo_map_gridsearch_ct"

plot_convergence_metrics = True

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

img_size = 256

#%% lidc test data set
problem = "Tomography"
save_dir = f'../datasets/{problem}'

data_test = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)]

(x, _) = next(iter(data_test[0]))
x = x.unsqueeze(0).to(device)
plot([x])

#%%

# #%% test with set3c
# val_transform = T.Compose(
#     [T.CenterCrop(img_size), T.ToTensor()]
# )
# dataset = load_dataset("set3c", transform=val_transform)

# x = dataset[2][0].unsqueeze(0).unsqueeze(0).to(device)


# %%
# Generate a physics operator for CT

poisson_level = 5  # Poisson noise level for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images

num_angles = 360
noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0/poisson_level))
physics = dinv.physics.Tomography(angles=num_angles,
    img_width=img_size,
    device=device, 
    noise_model=noise_model,
)


#%%

# Apply the degradation to the image
y = physics(x).clamp(1e-4,None)

plot([x,y, physics.A_dagger(y)])



# %%
# Select the data fidelity term
data_fidelity = PoissonLikelihood(gain=1.0/poisson_level, bkg=1e-8)

#%% Setup the reconstruction method
#
# Prox Drunet

# The GSPnP prior corresponds to a RED prior with an explicit `g`.
# We thus write a class that inherits from RED for this custom prior.
class GSPnP(dinv.optim.prior.RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


def custom_output(X):
    return X["est"][1]

# pretrained_path = "..\\..\\..\\bregman_sampling\\BregmanPnP\\GS_denoising\\ckpts\\Prox-DRUNet.ckpt"
    
# filepath = hf_hub_download(
#     repo_id="deepinv/gradientstep",     
#     filename="Prox-DRUNet.ckpt"
# )
# hf_path = "https://huggingface.co/deepinv/gradientstep/resolve/main/Prox-DRUNet.ckpt"
# Specify the Denoising prior
# prior = GSPnP(denoiser=dinv.models.GSDRUNet(act_mode='s', 
#                                            pretrained=filepath).to(device))

# pretrained = "../datasets/Tomographysup/25-08-25-15_19_57/ckp_best.pth.tar" 

#Artifact removal doesn't work here
# denoiser = dinv.models.ArtifactRemoval(
#     backbone_net=dinv.models.UNet(in_channels=1, out_channels=1, 
#             scales=4, bias=False, batch_norm=False))
# denoiser.load_state_dict(torch.load(pretrained, map_location=device)['state_dict'])
# denoiser.eval()

model_gsdrunet = dinv.models.GSDRUNet(
        in_channels=1,
        out_channels=1,
        device=device,
        pretrained=Path(
            "ct-drunet-weights/ckp_532.pth.tar"
        ),
    )

prior = dinv.optim.ScorePrior(
    denoiser=dinv.models.UNet(in_channels=1, 
                              out_channels=1, 
                              scales=4, 
                              bias=False, 
                              batch_norm=False)

).to(device)

trained_path = "./datasets/Tomographysup_ct_denoising/ckp_best.pth.tar"
ckpt = torch.load(trained_path, map_location=device)
prior.denoiser.load_state_dict(ckpt["state_dict"])



# prior = GSPnP(denoiser=model_gsdrunet.to(device))
# prior = dinv.optim.ScorePrior(denoiser=model_gsdrunet.to(device)).to(device)

# ram_model = dinv.models.RAM(device=device, pretrained=True)
# t =ram_model(y, physics)
# print(t)
# #%%
# prior =dinv.optim.PnP(denoiser=ram_model).to(device)

#%%

max_iter = 20
denoiser_factors = [1, 0.5, 0.1]
lamb_list = [5, 1e1, 30]
stepsize_list = [1e-4, 1e-4, 1e-4] 
#Lipschitz constant 
# stepsize = 1/(torch.pi / (2 * num_angles))

for i in range(len(lamb_list)):
    for j in range(len(denoiser_factors)):
        lamb = lamb_list[i] 
        stepsize = stepsize_list[i]
        sigma_denoiser = denoiser_factors[j] * (25/255.0)
        print(f"=== Running for lamb = {lamb} and sigma_denoiser factor = { denoiser_factors[j]} ===")

        params_algo = {
            "stepsize": stepsize,
            "g_param": sigma_denoiser,
            "lambda": lamb,
        }


        # instantiate the algorithm class to solve the IP problem.
        model = optim_builder(
            iteration="PGD",
            prior=prior,
            g_first=True,
            data_fidelity=data_fidelity,
            params_algo=params_algo,
            early_stop=True,
            max_iter=max_iter,
            crit_conv="residual", # "cost" only works for GSPnP
            thres_conv=1e-5,
            backtracking=True,
            # get_output=custom_output,
            verbose=False,
            custom_init=lambda observation, physics: {
            "est": (physics.A_dagger(observation), physics.A_dagger(observation))
            },  # initialize the optimization with FBP reconstruction
        )
        # model.eval()


        #run the model on the problem.
        with torch.no_grad():
            x_model, metrics = model(
                y, physics, x_gt=x, compute_metrics=True
            )  # reconstruction with PGD algorithm

        print("range of x_model: ", x_model.min().item(), x_model.max().item())

        print(f"initial PSNR: {dinv.metric.PSNR()(x, physics.A_dagger(y).clamp(0,1)).item():.2f} dB")
        print(f"reconstruction PSNR: {dinv.metric.PSNR()(x, x_model.clamp(0,1)).item():.2f} dB")

        print(f"reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model.clamp(0,1)).item():.3f}")


        # plot images. Images are saved in RESULTS_DIR.
        imgs = [y, x, physics.A_dagger(y), x_model]
        plot(
            imgs,
            titles=["Input", "GT", "FBP", "recon"],
            save_dir=RESULTS_DIR,
            rescale_mode="min_max"
        )

        # plot convergence curves
        if plot_convergence_metrics:
            plot_curves(metrics)



# %% plot inset
    
dinv.utils.plot_inset( imgs,
    titles=["Observation", "GT", "FBP", "Recon"],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "inset.png",
    )
# %%

dinv.utils.plot_inset( [y],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "obs_inset.png",
    )

dinv.utils.plot_inset( [x],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "gt_inset.png",
    )

dinv.utils.plot_inset( [x_model],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "recon_inset.png",
    )
# %%
