import torch
from torch.autograd.functional import jvp, vjp  # requires reasonably recent PyTorch
import deepinv as dinv
from pathlib import Path


# Fallback finite-difference Jv if jvp/vjp not available:
def jvp_fd(model, x, v, eps=1e-4):
    # directional derivative approx: (D(x+eps v)-D(x)) / eps
    return (model(x + eps * v) - model(x)) / eps

def est_local_lipschitz_fd(model, x, iters=30, eps=1e-4):
    model.eval()
    v = torch.randn_like(x)
    v = v / (v.norm() + 1e-12)
    last_lambda = 0.0
    for _ in range(iters):
        Jv = jvp_fd(model, x, v, eps=eps)
        # approximate J^T (Jv) by finite-difference of the scalar inner product:
        # w_i ≈ ( (D(x+eps e_i)·(Jv)) - (D(x)·(Jv)) ) / eps  -- expensive if done directly.
        # Instead approximate J^T (Jv) ≈ (grad wrt x of <D(x), Jv>) using autograd:
        x_req = x.clone().detach().requires_grad_(True)
        y = model(x_req)
        s = (y * Jv).sum()
        JtJv = torch.autograd.grad(s, x_req)[0]
        w = JtJv
        norm_w = w.norm()
        if norm_w.item() == 0:
            return 0.0, v
        v = w / norm_w
        lam = (Jv.view(-1).dot(Jv.view(-1))).item()
        if abs(lam - last_lambda) < 1e-6 * max(1.0, last_lambda):
            break
        last_lambda = lam
    return float(last_lambda**0.5), v.detach()

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


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

#%% model 2
#collection of paths 
net_path = "../../../bregman_sampling/Prox_GSPnP/GS_denoising/ckpts/test/epoch=808-step=504816.ckpt"
#net_path = "../../../bregman_sampling/Prox_GSPnP/GS_denoising/ckpts/test_reg/epoch=9-step=6240.ckpt"


#soon to be proxdrunet
model_gsdrunet = dinv.models.GSDRUNet(
    in_channels=1,
    out_channels=1,
    act_mode='s',
    device=device,
    pretrained=Path(
        #"datasets/ct_drunet/epoch=69-step=43680.ckpt"
        # "datasets/ct_drunet/epoch=77-step=48672.ckpt"
        #"datasets/ct_drunet/epoch=113-step=71136.ckpt"
        # "../../../bregman_sampling/Prox_GSPnP/GS_denoising/ckpts/test/epoch=808-step=504816.ckpt"
        net_path
    ),
)

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


prior2 = GSPnP(denoiser=model_gsdrunet.to(device))



BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results" / "demo_map_gridsearch_ct"


img_size = 256

#%% lidc test data set
problem = "Tomography"
save_dir = f'datasets/{problem}'

data_test = [dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=False)]
iterator = iter(data_test[0])

#40
#3

for i in range(3):
    (x, _) = next(iterator)

x = x.unsqueeze(0).to(device)

# model is your denoiser (e.g., UNet) mapping (1,C,H,W)->(1,C,H,W)
# x = torch.randn(1,1,128,128, device='cuda') * 0.1  # pick a representative input
#sigma, v = est_local_lipschitz_fd(prior.denoiser, x.to(device), iters=40)
#print("Estimated local Lipschitz (sigma_max):", sigma)

sigma2, v = est_local_lipschitz_fd(prior2.denoiser, x.to(device), iters=40)
print("Estimated local Lipschitz 2 (sigma_max):", sigma2)

# %%
