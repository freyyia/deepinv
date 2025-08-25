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
RESULTS_DIR = BASE_DIR / "results" / "demo_map_gridsearch_ct"
DEG_DIR = BASE_DIR / "degradations"

plot_convergence_metrics = True


# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the Set3C dataset and a motion blur kernel from
# `Levin et al. (2009) <https://ieeexplore.ieee.org/abstract/document/5206815/>`_.
#

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Set up the variable to fetch dataset and operators.
# dataset_name = "set3c"
# img_size = 64 if torch.cuda.is_available() else 64
# val_transform = transforms.Compose(
#     [transforms.CenterCrop(img_size), transforms.ToTensor()]
# )

# dataset = load_dataset(dataset_name, transform=val_transform)
img_size = 256
problem = 'Tomography'
save_dir = f'../datasets/{problem}'
data_test = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)

#get item via (x,y) = test_dataloader.__getitem__(idx)


# %%
# Generate a physics operator for CT

poisson_level = 20  # Poisson noise level for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images

mu = 1 / 50.0 * (362.0 / img_size)
num_angles = 360
# angles = torch.linspace(0, 360, steps=num_angles)
noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0/poisson_level))
physics = dinv.physics.Tomography(angles=num_angles,
    img_width=img_size,
    #fbp_interpolate_boundary=True, 
    device=device, 
    noise_model=noise_model
)


# Select the first image from the dataset
# x = dataset[2][0].mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)
(x_t,y_t) = data_test.__getitem__(0)
x = x_t.unsqueeze(0).to(device)
y = y_t.unsqueeze(0).to(device)

# Apply the degradation to the image
y = physics(x)


# %%
# Select the data fidelity term
data_fidelity = PoissonLikelihood(gain=1.0/poisson_level, bkg=1e-8)

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
# prior = GSPnP(denoiser=dinv.models.GSDRUNet(act_mode='s', 
#                                            pretrained=filepath).to(device))

pretrained = "../datasets/Tomographysup/25-08-25-15_19_57/ckp_best.pth.tar" 

#Artifact removal doesn't work here
# denoiser = dinv.models.ArtifactRemoval(
#     backbone_net=dinv.models.UNet(in_channels=1, out_channels=1, 
#             scales=4, bias=False, batch_norm=False))
# denoiser.load_state_dict(torch.load(pretrained, map_location=device)['state_dict'])
# denoiser.eval()

# prior = dinv.optim.ScorePrior(
#     denoiser=dinv.models.DnCNN(in_channels=1, out_channels=1, pretrained="download_lipschitz")
# ).to(device)

prior = dinv.optim.prior.TVPrior(n_it_max=20).to(device)



#%%
# we want to output the intermediate PGD update to finish with a denoising step.
def custom_output(X):
    return X["est"][1]

max_iter = 5
multiple_list = [1]
lamb_list = [1,5]
stepsize_list = [0.0001] 

for i in range(len(lamb_list)):
    for j in range(len(multiple_list)):
        lamb = lamb_list[i] 
        stepsize = 1
        sigma_denoiser = 2/255.0 #multiple_list[j] * (25/255.0)  
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
            custom_init=lambda observation, physics: {
            "est": (physics.A_dagger(observation), physics.A_dagger(observation))
            },  # initialize the optimization with FBP reconstruction
        )

        #run the model on the problem.
        with torch.no_grad():
            x_model_proxdrunet, metrics_proxdrunet = model(
                y, physics, x_gt=x, compute_metrics=True
            )  # reconstruction with PGD algorithm
        print(f"Prox-DRUNet reconstruction PSNR: {dinv.metric.PSNR()(x, x_model_proxdrunet.clamp(0,1)).item():.2f} dB")

        print(f"Prox-DRUNet reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model_proxdrunet.clamp(0,1)).item():.3f}")


        # plot images. Images are saved in RESULTS_DIR.
        imgs = [y, x,  x_model_proxdrunet]
        plot(
            imgs,
            titles=["Input", "GT", "Proxdrunet"],
            save_dir=RESULTS_DIR,
            #rescale_mode="clip"
        )

        # plot convergence curves
        if plot_convergence_metrics:
            plot_curves(metrics_proxdrunet)



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
