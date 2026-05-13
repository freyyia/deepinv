from __future__ import annotations
import torch
from deepinv.physics import Physics
from deepinv.optim.prior import Prior
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim import BurgEntropy


class MLAIterator(SamplingIterator):
    r"""
    Single iteration of the Mirror Langevin Algorithm (MLA).

    Designed for Bayesian inverse problems with positive-valued images, using the
    Burg entropy as a mirror map to ensure iterates remain strictly positive.

    .. warning::
        MLA requires strictly positive inputs. Ensure measurements and initialisation
        satisfy :math:`x > 0`.

    :param dict algo_params: Dictionary containing the algorithm parameters (see table below)

    .. list-table::
       :widths: 15 10 75
       :header-rows: 1

       * - Parameter
         - Type
         - Description
       * - step_size
         - float
         - Step size :math:`\eta > 0` of the algorithm
       * - alpha
         - float
         - Regularization parameter :math:`\alpha` (default: 1.0)
       * - sigma
         - float
         - Noise level for the score prior denoiser. A larger value results in a more regularized reconstruction
    """

    def __init__(self, algo_params: dict[str, float], clip: tuple[float, float] | None = None, **kwargs):
        super().__init__(algo_params)

        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "sigma" not in algo_params:
            missing_params.append("sigma")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for MLA: {', '.join(missing_params)}"
            )

        self.potential = BurgEntropy()
        # Only the upper bound is needed: the Burg mirror map keeps iterates strictly
        # positive automatically, so only clip(x, None, upper) is applied.
        self.clip_upper = clip[1] if clip is not None else None

    def forward(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        iteration: int,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        r"""
        Performs a single MLA sampling step.

        :param dict X: Dictionary containing the current state :math:`x_t`.
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param Prior cur_prior: Prior
        :param int iteration: Current iteration number

        :return: Dictionary ``{"x": x}`` containing the next state :math:`x_{t+1}` in the Markov chain.
        :rtype: dict
        """
        x = X["x"]
        noise = torch.randn_like(x) * torch.sqrt(2 * self.algo_params["step_size"] * self.potential.hessian(x))
        yk1 = self.potential.grad(x)
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior = (
            -cur_prior.grad(x, self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        yk2 = yk1 + self.algo_params["step_size"] * (lhood + lprior) + noise
        xk = self.potential.grad_conj(yk2).clamp(1e-6, None)
        if self.clip_upper is not None:
            xk = xk.clamp(None, self.clip_upper)
        return {"x": xk}


class MLA3DIterator(SamplingIterator):
    r"""
    Single iteration of the Mirror Langevin Algorithm (MLA).

    Designed for Bayesian inverse problems with positive-valued images, using the
    Burg entropy as a mirror map to ensure iterates remain strictly positive. 
    
    This version is meant to support two planes data.

    .. warning::
        MLA requires strictly positive inputs. Ensure measurements and initialisation
        satisfy :math:`x > 0`.

    :param dict algo_params: Dictionary containing the algorithm parameters (see table below)

    .. list-table::
       :widths: 15 10 75
       :header-rows: 1

       * - Parameter
         - Type
         - Description
       * - step_size
         - float
         - Step size :math:`\eta > 0` of the algorithm
       * - alpha
         - float
         - Regularization parameter :math:`\alpha` (default: 1.0)
       * - sigma
         - float
         - Noise level for the score prior denoiser. A larger value results in a more regularized reconstruction
    """

    def __init__(self, algo_params: dict[str, float], clip: tuple[float, float] | None = None, **kwargs):
        super().__init__(algo_params)

        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "sigma" not in algo_params:
            missing_params.append("sigma")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for MLA: {', '.join(missing_params)}"
            )

        self.potential = BurgEntropy()
        # Only the upper bound is needed: the Burg mirror map keeps iterates strictly
        # positive automatically, so only clip(x, None, upper) is applied.
        self.clip_upper = clip[1] if clip is not None else None

    def forward(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        iteration: int,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        r"""
        Performs a single MLA sampling step.

        :param dict X: Dictionary containing the current state :math:`x_t`.
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param Prior cur_prior: Prior
        :param int iteration: Current iteration number

        :return: Dictionary ``{"x": x}`` containing the next state :math:`x_{t+1}` in the Markov chain.
        :rtype: dict
        """
        x = X["x"]
        # num_planes = x.shape[1]
        # x_prior = x if num_planes == 1 else x[:,1:2]
        noise = torch.randn_like(x) * torch.sqrt(2 * self.algo_params["step_size"] * self.potential.hessian(x))
        yk1 = self.potential.grad(x)
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior_for = (
            -cur_prior.grad(x[:,1:2], self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        lprior_back = (
            -cur_prior.grad(x[:,0:1], self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        # lprior_empty = torch.zeros_like(lprior)
        lprior_final = torch.cat([lprior_back, lprior_for], dim=1)
        yk2 = yk1 + self.algo_params["step_size"] * (lhood + lprior_final) + noise
        xk = self.potential.grad_conj(yk2).clamp(1e-6, None)
        if self.clip_upper is not None:
            xk = xk.clamp(None, self.clip_upper)
        return {"x": xk}