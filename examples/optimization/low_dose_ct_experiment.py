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
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

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
    parser.add_argument("--im_idx", type=int, default=0)
    parser.add_argument("--prior_type", type=int, default=1)
    parser.add_argument("--method", type=str, default="SKROCK")
    parser.add_argument("--poisson_level", type=float, default=10)
    parser.add_argument("--regularization", type=float, default=4)
    parser.add_argument("--step_size", type=float, default=5e-5)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--inset_loc", type=ast.literal_eval, default=(0.52, 0.55))

    return parser.parse_args()

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



def main():
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


    # Apply the degradation to the image
    y = physics(x).clamp(1e-4,None)
    print("y min/max: ", y.min().item(), y.max().item())
    print("x min/max: ", x.min().item(), x.max().item())
    print("FBP min/max: ", physics.A_dagger(y).min().item(), physics.A_dagger(y).max().item())

    plot([x,y, physics.A_dagger(y)])

    image_mean = args.poisson_level*torch.mean(x)
    beta = image_mean * 0.02 

    data_fidelity = PoissonLikelihood(gain=1.0/args.poisson_level, bkg=beta.item())

    if not args.prior_type == 2:
        if args.prior_type == 0:
            net_path = "../../../bregman_sampling/Prox_GSPnP/GS_denoising/ckpts/test/epoch=808-step=504816.ckpt"
        elif args.prior_type == 1:
            net_path = "../../../bregman_sampling/Prox_GSPnP/GS_denoising/ckpts/test_reg/epoch=26-step=16848.ckpt"

        model_gsdrunet = dinv.models.GSDRUNet(
            in_channels=1,
            out_channels=1,
            act_mode='s',
            device=device,
            pretrained=Path(
                net_path
            )
        )
        #prior = GSPnP(denoiser=model_gsdrunet.to(device))
        prior = dinv.optim.ScorePrior(denoiser=model_gsdrunet.to(device)).to(device)



    if args.prior_type == 2:
        prior = dinv.optim.ScorePrior(
            denoiser=dinv.models.UNet(in_channels=1, 
                                    out_channels=1, 
                                    scales=4, 
                                    bias=False, 
                                    batch_norm=False)

        ).to(device)

        net_path = "datasets/Tomographysup_ct_denoising/25-08-27-15:53:54/ckp_best.pth.tar"
        ckpt = torch.load(net_path, map_location=device)
        prior.denoiser.load_state_dict(ckpt["state_dict"])





    # the operator 


    L = 1.0
    AAT_norm = 88984.89
    L_y =  args.poisson_level**2*(torch.max(y)/beta**2)*AAT_norm
    eps = (25/255)**2

    delta_max = 1.0/(L/eps+L_y)
    print("Stepsize: ", delta_max)
    delta_frac = 1
    delta = delta_max*delta_frac


    # Create the MCMC sampler
    # --------------------------------------------------------------
    #


    iterations = int(args.iterations) if torch.cuda.is_available() else 10
    params = {
        "step_size": args.step_size,
        "alpha": args.regularization,
        "sigma": 1*(20/255.0),
        "eta"  : 0.05,
        "inner_iter": 10,
        "method" : args.method,
        "network": net_path,

    }
    RESULTS_DIR = BASE_DIR / "results" / (
        f"low_dose_ct_{params['method']}_im_{args.im_idx}_{prior_types[args.prior_type]}_pl_{args.poisson_level}_reg_{args.regularization}_step_{args.step_size}_it_{args.iterations}"
    )



    #%% init wandb
    project = "sampling_ct"
    use_wandb = True
    if use_wandb:
        wandb.init(entity='bloom', project=project, config=params, save_code=True)
    else:
        wandb.init(mode="disabled")
    # #%% log measurement
    wandb.log({"Observation" : wandb.Image((y/torch.max(y)).cpu().squeeze(), caption="Observation")})
    # log ground truth
    wandb.log({"Ground truth" : wandb.Image((x).cpu().squeeze(), caption="Ground truth")})


    #%% define the callback for wandb logging
    def call(X, statistics, iter, **kwargs):
        psnr_log = dinv.metric.PSNR()(x, statistics[0].mean().clamp(0,1)).item()
        lpips_log = dinv.metric.LPIPS(device=device)(x, statistics[0].mean().clamp(0,1)).item()
        print(f"PSNR: {psnr_log:.2f} dB")
        wandb.log({"PSNR" : psnr_log,
                   "LPIPS": lpips_log,
                "Variance" : wandb.Image(statistics[0].var().sqrt().cpu().squeeze(),
                                            caption="Variance"),
                "Posterior mean" : wandb.Image(statistics[0].mean().cpu().squeeze(),
                                            caption="Mean")},
                step=iter+1)
        
    print(args.method)

    f = dinv.sampling.sampling_builder(
        iterator=str(args.method).upper(),
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=iterations,
        params_algo=params,
        thinning=10,
        verbose=True,
        clip=[0,1],
        callback=call,
    )

    #% run the sampler
    mean, var = f.sample(y, physics, x_init=physics.A_dagger(y).clamp(1e-4,1))


    print("range of mean: ", mean.min().item(), mean.max().item())

    print(f"initial PSNR: {dinv.metric.PSNR()(x, physics.A_dagger(y).clamp(0,1)).item():.2f} dB")
    print(f"reconstruction PSNR: {dinv.metric.PSNR()(x, mean.clamp(0,1)).item():.2f} dB")

    print(f"reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, mean.clamp(0,1)).item():.3f}")


    # plot images. Images are saved in RESULTS_DIR.
    imgs = [y, x, physics.A_dagger(y), mean]
    plot(
        imgs,
        titles=["Input", "GT", "FBP", "recon"],
        save_dir=RESULTS_DIR,
        rescale_mode="min_max",
        #rescale_mode="clip"
    )

    plot([torch.sqrt(var)], titles=["STD"], save_dir=RESULTS_DIR)

    torch.save(mean.cpu(), RESULTS_DIR / "mean.pt")
    torch.save(torch.sqrt(var).cpu(), RESULTS_DIR / "std.pt")


    # import matplotlib.pyplot as plt
    # plt.imshow(torch.sqrt(var).cpu().squeeze(), vmax=0.1207, cmap='gray')
    # plt.colorbar()

    # %% plot inset
        
    dinv.utils.plot_inset( imgs,
        titles=["Observation", "GT", "FBP", "Recon"],
        extract_loc=args.inset_loc,
        inset_loc=(0.0, 0.6),
        save_fn = RESULTS_DIR / "inset.png",
        )
    # %%

    dinv.utils.plot_inset( [y],
        titles=[""],
        extract_loc=args.inset_loc,
        inset_loc=(0.0, 0.6),
        save_fn = RESULTS_DIR / "obs_inset.png",
        )

    dinv.utils.plot_inset( [physics.A_dagger(y)],
        titles=[""],
        extract_loc=args.inset_loc,
        inset_loc=(0.0, 0.6),
        save_fn = RESULTS_DIR / "fbp_inset.png",
        )

    dinv.utils.plot_inset( [x],
    titles=[""],
    extract_loc=args.inset_loc,
    inset_loc=(0.0, 0.6),
    save_fn = RESULTS_DIR / "gt_inset.png",
    )

    dinv.utils.plot_inset( [mean],
        titles=[""],
        extract_loc=args.inset_loc,
        inset_loc=(0.0, 0.6),
        save_fn = RESULTS_DIR / "recon_inset.png",
        )


if __name__ == "__main__":
    main()