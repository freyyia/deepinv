import torch
from torch import Tensor, Tuple


class Welford:
    r"""
    Welford's algorithm :footcite:t:`welford1962note`for calculating mean and variance.

    :param torch.Tensor x: Initial sample (defines shape and device).
    :param bool track_quantiles: If ``True``, maintain streaming quantile
        estimates for coverage map computation. Default: ``False``
        (preserves original behaviour).
    :param float alpha_ci: Significance level for credible intervals.
        Default: ``0.05`` (95% CI).
    :param float rm_c: Robbins-Monro learning rate scale. Larger values
        give faster adaptation but noisier early estimates. Default: ``2.0``.
    :param float rm_k0: Robbins-Monro offset for step-size stability in
        early iterations. Default: ``20.0``.

    The core Welford recurrence is numerically stable and O(1) per update:
 
    .. math::
 
        \delta   &= x - \mu_{k-1} \\
        \mu_k    &= \mu_{k-1} + \delta / k \\
        M_{2,k}  &= M_{2,k-1} + \delta \cdot (x - \mu_k)
 
    When ``track_quantiles=True``, two additional per-pixel quantile
    estimates are maintained via scale-adapted Robbins-Monro stochastic
    approximation, enabling distribution-free credible intervals:
 
    .. math::
 
        q_p \leftarrow q_p
            + \frac{c \cdot \hat\sigma}{k + k_0}
              \bigl(p - \mathbf{1}\{x \le q_p\}\bigr)
 
    This avoids storing all posterior samples and allows coverage maps
    to be computed on-the-fly during MCMC sampling.

    Basic usage (backward-compatible)::
 
        >>> w = Welford(x0)
        >>> for x in posterior_samples:
        ...     w.update(x)
        >>> posterior_mean = w.mean()
        >>> posterior_var  = w.var()
 
    With online coverage maps::
 
        >>> w = Welford(x0, track_quantiles=True, alpha_ci=0.05)
        >>> for x in posterior_samples:
        ...     w.update(x)
        >>> lower, upper = w.ci()           # credible intervals
        >>> cov_map = w.coverage(x_true)    # binary coverage map
        >>> unc_map = w.std()               # uncertainty map

    """

    def __init__(self, x: Tensor, 
                 track_quantiles: bool = False,
                 alpha_ci: float = 0.05,
                 rm_c: float = 2.0,
                 rm_k0: float = 20.0,):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

        # Quantile tracking
        self.track_quantiles = track_quantiles
        if track_quantiles:
            self.alpha_ci = alpha_ci
            self.p_lo = alpha_ci / 2.0
            self.p_hi = 1.0 - alpha_ci / 2.0
            self.rm_c = rm_c
            self.rm_k0 = rm_k0
            self.q_lo = x.clone()
            self.q_hi = x.clone()

    def update(self, x: Tensor):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

        if self.track_quantiles:
            scale = self.var().sqrt().clamp(min=1e-8)
            eta = self.rm_c * scale / (self.k + self.rm_k0)
            self.q_lo = self.q_lo + eta * (self.p_lo - (x <= self.q_lo).float())
            self.q_hi = self.q_hi + eta * (self.p_hi - (x <= self.q_hi).float())

    def mean(self) -> Tensor:
        return self.M

    def var(self) -> Tensor:
        if self.k > 1:
            return self.S / (self.k - 1)
        else:
            return self.S
        
    def ci(self, method: str = "auto") -> Tuple[Tensor, Tensor]:
        r"""
        Pixel-wise credible intervals from the running statistics.
 
        Three methods are available:
 
        - ``"gaussian"``: Symmetric intervals
          :math:`\mu \pm z_{1-\alpha/2} \cdot \sigma`.
          Fast and stable, but assumes approximate normality of
          the marginal posteriors.
        - ``"quantile"``: Distribution-free intervals from the
          streaming Robbins-Monro quantile estimates.
          Requires ``track_quantiles=True``.
        - ``"auto"`` (default): Uses ``"quantile"`` if quantile
          tracking is enabled, otherwise ``"gaussian"``.
 
        :param str method: ``"auto"``, ``"gaussian"``, or ``"quantile"``.
        :return: ``(lower, upper)`` tensors with same shape as samples.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if method == "auto":
            method = "quantile" if self.track_quantiles else "gaussian"
 
        if method == "gaussian":
            alpha = self.alpha_ci if self.track_quantiles else 0.05
            z = torch.distributions.Normal(0, 1).icdf(
                torch.tensor(1.0 - alpha / 2.0)
            ).item()
            s = self.var().sqrt()
            return self.M - z * s, self.M + z * s
 
        elif method == "quantile":
            if not self.track_quantiles:
                raise RuntimeError(
                    "Quantile tracking not enabled. "
                    "Pass track_quantiles=True to __init__."
                )
            return self.q_lo, self.q_hi
 
        else:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Use 'auto', 'gaussian', or 'quantile'.")
 
    def ci_width(self, **kwargs) -> Tensor:
        r"""Width of the credible interval at each pixel."""
        lo, hi = self.ci(**kwargs)
        return hi - lo
 
    def coverage(self, x_true: Tensor, **kwargs) -> Tensor:
        r"""
        Binary pixel-wise coverage map.
 
        For each pixel :math:`j`, checks whether the ground truth
        :math:`x^*_j` lies inside the credible interval:
 
        .. math::
 
            \text{coverage}_j =
              \mathbf{1}\bigl\{
                q_{\alpha/2,j} \le x^*_j \le q_{1-\alpha/2,j}
              \bigr\}
 
        A well-calibrated posterior should yield average coverage
        close to :math:`1 - \alpha`.
 
        :param torch.Tensor x_true: Ground truth image.
        :param kwargs: Forwarded to :meth:`ci` (e.g. ``method``).
        :return: Binary tensor (1 = covered, 0 = not covered).
        :rtype: torch.Tensor
        """
        lo, hi = self.ci(**kwargs)
        return ((x_true >= lo) & (x_true <= hi)).float()
 
    def avg_coverage(self, x_true: Tensor, **kwargs) -> float:
        r"""Scalar average coverage across all pixels."""
        return self.coverage(x_true, **kwargs).mean().item()


def refl_projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.clamp(x, min=lower, max=upper)
