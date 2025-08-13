

import deepinv as dinv
from pathlib import Path
import torch
from torchvision import transforms

from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils.plotting import plot, plot_curves, plot_inset
from huggingface_hub import hf_hub_download
from deepinv.utils.parameters import get_GSPnP_params


import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
})


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results" / "demo_map_gridsearch_inpainting"
DEG_DIR = BASE_DIR / "degradations"

plot_convergence_metrics = True


# %%
# Load base image datasets and degradation operators.


# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Set up the variable to fetch dataset and operators.
dataset_name = "set3c"
img_size = 64 if torch.cuda.is_available() else 64
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)

dataset = load_dataset(dataset_name, transform=val_transform)


# %%
# Generate a dataset of blurred images and load it.
# --------------------------------------------------------------------------------
# We use the BlurFFT class from the physics module to generate a dataset of blurred images.


poisson_level = 20  # Poisson noise level for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images

physics = dinv.physics.Inpainting(
    tensor_size=(n_channels, img_size, img_size),
    mask=0.7,
    device=device,
    noise_model=dinv.physics.Denoising(dinv.physics.PoissonNoise(torch.tensor(1.0/poisson_level))),
)

# Select the first image from the dataset
# x = dataset[2][0].mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)
x = dataset[2][0].unsqueeze(0).to(device)


# Apply the degradation to the image
y = physics(x)


# %%
# Select the data fidelity term
data_fidelity = PoissonLikelihood(gain=1.0/poisson_level, bkg=1e-4)

#%% Setup the second reconstruction method
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



# pretrained_path = "..\\..\\..\\bregman_sampling\\BregmanPnP\\GS_denoising\\ckpts\\Prox-DRUNet.ckpt"
    
filepath = hf_hub_download(
    repo_id="deepinv/gradientstep",     
    filename="Prox-DRUNet.ckpt"
)
# hf_path = "https://huggingface.co/deepinv/gradientstep/resolve/main/Prox-DRUNet.ckpt"
# Specify the Denoising prior
prior = GSPnP(denoiser=dinv.models.GSDRUNet(act_mode='s', 
                                           pretrained=filepath).to(device))

#%%

# lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
#     problem="deblur", noise_level_img=20/255.0
# )


#%%

# we want to output the intermediate PGD update to finish with a denoising step.
def custom_output(X):
    return X["est"][1]

max_iter = 600
# sigma_denoiser = 1.8 * 25/255.0
multiple_list = [1.5]
lamb_list = [0.1]#[0.25][0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
stepsize_list = [1] 

for i in range(len(lamb_list)):
    for j in range(len(multiple_list)):
        lamb = lamb_list[i] 
        stepsize = 1
        sigma_denoiser = multiple_list[j] * (25/255.0)  
        print(f"=== Running for lamb = {lamb} and sigma_denoiser factor = { multiple_list[j]} ===")

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
            early_stop=False,
            max_iter=max_iter,
            crit_conv="cost",
            thres_conv=1e-5,
            backtracking=True,
            get_output=custom_output,
            verbose=False,
        )

        #run the model on the problem.
        with torch.no_grad():
            x_model_proxdrunet, metrics_proxdrunet = model(
                y, physics, x_gt=x, compute_metrics=True
            )  # reconstruction with PGD algorithm
        print(f"Prox-DRUNet reconstruction PSNR: {dinv.metric.PSNR()(x, x_model_proxdrunet.clamp(0,1)).item():.2f} dB")

        print(f"Prox-DRUNet reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model_proxdrunet.clamp(0,1)).item():.3f}")


        # plot images. Images are saved in RESULTS_DIR.
        # imgs = [y, x, x_lin, x_model]
        imgs = [y, x,  x_model_proxdrunet]
        plot(
            imgs,
            titles=["Input", "GT", "Proxdrunet"],
            save_dir=RESULTS_DIR,
            rescale_mode="clip"
        )

        # plot convergence curves
        if plot_convergence_metrics:
            plot_curves(metrics_proxdrunet)





# %



# %% plot inset
    
dinv.utils.plot_inset( imgs,
    titles=["Observation", "GT", "Prox-DRUNet"],
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

dinv.utils.plot_inset( [x_model_proxdrunet],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "proxdrunet_inset.png",
    )
# %%
