import numpy as np
from scipy.ndimage import gaussian_filter, shift
import matplotlib.pyplot as plt

def gaussian_kernel(size=21, sigma=3):
    """Generate a normalized 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def shifted_gaussian_kernels(size=21, sigma=3, shift_px=3):
    """Generate a 3x3 grid of Gaussian kernels with shifts."""
    base = gaussian_kernel(size, sigma)
    shifts = [(-shift_px, -shift_px), (-shift_px, 0), (-shift_px, shift_px),
              (0, -shift_px), (0, 0), (0, shift_px),
              (shift_px, -shift_px), (shift_px, 0), (shift_px, shift_px)]
    
    kernels = [shift(base, s, mode='nearest') for s in shifts]
    return np.array(kernels).reshape(3, 3, size, size)

# Example usage
kernels = shifted_gaussian_kernels(size=21, sigma=1.8, shift_px=2)

# Visualize
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i in range(3):
    for j in range(3):
        axes[i, j].imshow(kernels[i, j], cmap='viridis')
        axes[i, j].axis('off')
plt.suptitle("3x3 Gaussian Sensor Array")
plt.show()
