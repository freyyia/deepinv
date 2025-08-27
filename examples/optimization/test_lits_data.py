
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)


def plot_sample(array_list, color_map='nipy_spectral'):
    '''
    Plots a slice with all available annotations
    '''
    fig = plt.figure(figsize=(18, 15))

    plt.subplot(2, 4, 1) 
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    
    # plt.subplot(2, 4, 2)
    # plt.imshow(np.clip(array_list[0], -150, 250), cmap='bone')
    # plt.title('Windowed Image [-150,250]')
        
    # plt.subplot(2, 4, 3)
    # plt.imshow(np.clip(array_list[0], -50, 200), cmap='bone')
    # plt.title('Windowed Image [-50,200]')

    # plt.subplot(2, 4, 4)
    # plt.imshow(np.clip(array_list[0], 30, 150), cmap='bone')
    # plt.title('Windowed Image [30,150]')
    


    plt.tight_layout()  # Added to prevent overlapping titles
    plt.show()

# Read sample
sample = 63
sample_ct   = read_nii("../../../deepinv/examples/optimization/datasets/lits-test-subset/test-volume-" + str(sample) + ".nii")

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
    return img


temp = sample_ct[...,70]
temp_t = torch.from_numpy(np.ascontiguousarray(temp))
plot_sample([hu_normalize_resize(temp_t).squeeze()])

