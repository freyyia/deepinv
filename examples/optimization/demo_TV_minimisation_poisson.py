r"""
Image deblurring with Total-Variation (TV) prior
====================================================================================================

This example shows how to use a standard TV prior for image deblurring. The problem writes as :math:`y = Ax + \epsilon`
where :math:`A` is a convolutional operator and :math:`\epsilon` is the realization of some Gaussian noise. The goal is
to recover the original image :math:`x` from the blurred and noisy image :math:`y`. The TV prior is used to regularize
the problem.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torchvision import transforms

from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils.plotting import plot, plot_curves, plot_inset

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
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"


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
dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 64
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions
dataset = load_dataset(dataset_name, transform=val_transform)


# %%
# Generate a dataset of blurred images and load it.
# --------------------------------------------------------------------------------
# We use the BlurFFT class from the physics module to generate a dataset of blurred images.


poisson_level = 20  # Poisson noise level for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.Denoising(dinv.physics.PoissonNoise(torch.tensor(1.0/poisson_level))),
)

# Select the first image from the dataset
# x = dataset[2][0].mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)
x = dataset[2].unsqueeze(0).to(device)

#%%

# Apply the degradation to the image
y1 = physics(x[0,0].unsqueeze(0).unsqueeze(0))
y2 = physics(x[0,1].unsqueeze(0).unsqueeze(0))
y3 = physics(x[0,2].unsqueeze(0).unsqueeze(0))
y_comb = torch.cat((y1, y2,y3),dim=1)

# %%
# Exploring the total variation prior.
# ------------------------------------
#
# In this example, we will use the total variation prior, which can be done with the :class:`deepinv.optim.prior.Prior`
# class. The prior object represents the cost function of the prior (TV in this case), as well as convenient methods,
# such as its proximal operator :math:`\text{prox}_{\tau g}`.

# Set up the total variation prior
prior = dinv.optim.prior.TVPrior(n_it_max=2000)

# Compute the total variation prior cost
cost_tv = prior(y_comb).item()
print(f"Cost TV: g(y) = {cost_tv:.2f}")

# Apply the proximal operator of the TV prior
x_tv = prior.prox(y_comb, gamma=0.1)
cost_tv_prox = prior(x_tv).item()

# %%
# .. note::
#           The output of the proximity operator of TV is **not** the solution to our deblurring problem. It is only a
#           step towards the solution and is used in the proximal gradient descent algorithm to solve the inverse
#           problem.
#

# Plot the input and the output of the TV proximal operator
imgs = [y_comb, x_tv]
plot(
    imgs,
    titles=[f"Input, TV cost: {cost_tv:.2f}", f"Output, TV cost: {cost_tv_prox:.2f}"],
)


# %%
# Set up the optimization algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# The problem we want to minimize is the following:
#
# .. math::
#
#     \begin{equation*}
#     \underset{x}{\operatorname{min}} \,\, \frac{1}{2} \|Ax-y\|_2^2 + \lambda \|Dx\|_{1,2}(x),
#     \end{equation*}
#
#
# where :math:`1/2 \|A(x)-y\|_2^2` is the a data-fidelity term, :math:`\lambda \|Dx\|_{2,1}(x)` is the total variation (TV)
# norm of the image :math:`x`, and :math:`\lambda>0` is a regularisation parameters.
#
# We use a Proximal Gradient Descent (PGD) algorithm to solve the inverse problem.

# Select the data fidelity term
data_fidelity = PoissonLikelihood(gain=1.0/poisson_level)

# Specify the prior (we redefine it with a smaller number of iteration for faster computation)
prior = dinv.optim.prior.TVPrior(n_it_max=20)

# specify pnp prior instead of TV prior
# Specify the denoising prior
# prior = dinv.optim.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))

# Logging parameters
verbose = True
plot_convergence_metrics = (
    True  # compute performance and convergence metrics along the algorithm.
)
sigma_denoiser = 0.05  # noise level for the denoiser

# Algorithm parameters
stepsize = 1.0
lamb = 0.8e-2  # TV regularisation parameter
params_algo = {"stepsize": stepsize, "lambda": lamb, "g_param": sigma_denoiser}

max_iter = 300
early_stop = True

# Instantiate the algorithm class to solve the problem.
model = optim_builder(
    iteration="PGD",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    params_algo=params_algo,
)

# run the model on the problem.
with torch.no_grad():
    x_model_tv, metrics_tv = model(
        y_comb, physics, x_gt=x, compute_metrics=True
    )  # reconstruction with PGD algorithm


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



pretrained_path = "..\\..\\..\\bregman_sampling\\BregmanPnP\\GS_denoising\\ckpts\\Prox-DRUNet.ckpt"
# Specify the Denoising prior
prior = GSPnP(denoiser=dinv.models.GSDRUNet(act_mode='s', 
                                           pretrained=pretrained_path).to(device))

# GSDRUNet PSNR 21.78
# prior = GSPnP(denoiser=dinv.models.GSDRUNet(pretrained="download").to(device))

# working, psnr 21.96 but a slight checkerboard pattern
# DncNN / LMMO
# prior = dinv.optim.ScorePrior(
#     denoiser=dinv.models.DnCNN(pretrained="download_lipschitz")
# ).to(device)

# lamb, sigma_denoiser, stepsize, max_iter = dinv.utils.get_GSPnP_params(
#     problem="deblur", noise_level_img=20/255.0
# )
# max_iter = 300
# sigma_denoiser = 20/255.0
# lamb = 0.5
# stepsize = 0.0001 #1 / lamb

max_iter = 350
sigma_denoiser = 1 * 25/255.0 #1.8
lamb = 0.1
stepsize = 1

params_algo = {
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
}

# we want to output the intermediate PGD update to finish with a denoising step.
def custom_output(X):
    return X["est"][1]


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
        torch.cat((y1, y2,y3),dim=1), physics, x_gt=x, compute_metrics=True
    )  # reconstruction with PGD algorithm






# %%

model = dinv.optim.RidgeRegularizer(pretrained="../../deepinv/saved_model/weights.pt").to(device)

with torch.no_grad():
    recon1 = model.reconstruct(physics, y1, 0.1, 1.0)
    recon2 = model.reconstruct(physics, y2, 0.1, 1.0)
    recon3 = model.reconstruct(physics, y3, 0.1, 1.0)


recon = torch.cat((recon1, recon2, recon3), dim=1)
plot([x, torch.cat((y1, y2,y3),dim=1), 
      recon], titles=["ground truth", "observation", "reconstruction"])

# %%
# Evaluate the model on the problem and plot the results.
# --------------------------------------------------------------------
#
# The model returns the output and the metrics computed along the iterations.
# For computing PSNR, the ground truth image ``x_gt`` must be provided.


# x_lin = physics.A_adjoint(y)  # linear reconstruction with the adjoint operator


# compute PSNR
# print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")

#%%
print(f"WCRNN reconstruction PSNR: {dinv.metric.PSNR()(x, recon).item():.2f} dB")
print(f"TV reconstruction PSNR: {dinv.metric.PSNR()(x, x_model_tv).item():.2f} dB")
print(f"Prox-DRUNet reconstruction PSNR: {dinv.metric.PSNR()(x, x_model_proxdrunet).item():.2f} dB")

print(f"WCRNN reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, recon).item():.3f}")
print(f"TV reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model_tv).item():.3f}")
print(f"Prox-DRUNet reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model_proxdrunet).item():.3f}")


# plot images. Images are saved in RESULTS_DIR.
# imgs = [y, x, x_lin, x_model]
imgs = [torch.cat((y1, y2,y3),dim=1), x, x_model_tv,recon,  x_model_proxdrunet]
plot(
    imgs,
    titles=["Input", "GT", "TV", "WCRNN", "Proxdrunet"],
    save_dir=RESULTS_DIR / "demo_map_poisson",

)

# plot convergence curves
if plot_convergence_metrics:
    plot_curves(metrics_tv)
    plot_curves(metrics_proxdrunet)

# %% plot inset
    
dinv.utils.plot_inset( imgs,
    titles=["Observation", "GT", "TV", "WCRNN", "Prox-DRUNet"],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "inset.png",
    )
# %%
#[torch.cat((y1, y2,y3),dim=1), x, x_model_tv,recon,  x_model_proxdrunet]

dinv.utils.plot_inset( [torch.cat((y1, y2,y3),dim=1)],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "obs_inset.png",
    )

dinv.utils.plot_inset( [x],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "gt_inset.png",
    )
dinv.utils.plot_inset( [x_model_tv],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "tv_inset.png",
    )
dinv.utils.plot_inset([recon],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "wcrr_inset.png",
    )
dinv.utils.plot_inset( [x_model_proxdrunet],
    titles=[""],
    extract_loc=(0.47, 0.45),
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "demo_map_poisson" / "proxdrunet_inset.png",
    )
# %%
