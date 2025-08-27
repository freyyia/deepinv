import torch
import deepinv as dinv
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator 

def compute_largest_eigenvalue_ATA(A, im_size, max_iter=100, tol=1e-6, device="cpu"):
    """
    Computes the largest eigenvalue of the operator A.adjoint(A) using power iteration.

    :param deepinv.physics.LinearPhysics A: The linear operator.
    :param tuple im_size: The size of the input images (domain of A).
    :param int max_iter: The maximum number of iterations.
    :param float tol: The tolerance for convergence.
    :param str device: The device to run the computation on.
    :return: The largest eigenvalue of A^T*A.
    """
    # 1. Initialize a random vector in the image domain
    vec = torch.randn(im_size, device=device)
    vec = vec / torch.linalg.norm(vec)  # Normalize initial vector

    lambda_old = 0.0

    # 2. Iterate by applying the square operator A.adjoint(A)
    for i in range(max_iter):
        vec_new = physics.A_adjoint(physics.A(vec))

        # 3. Estimate the eigenvalue using the Rayleigh quotient
        # lambda = (vec^T * vec_new) / (vec^T * vec)
        lambda_new = torch.dot(vec.flatten(), vec_new.flatten()) / torch.dot(vec.flatten(), vec.flatten())

        # 4. Normalize the new vector for the next iteration
        vec = vec_new / torch.linalg.norm(vec_new)

        # 5. Check for convergence
        if torch.abs(lambda_new - lambda_old) / (lambda_new + 1e-9) < tol:
            print(f"Converged at iteration {i+1}")
            break
        lambda_old = lambda_new

    return lambda_new.item()


#%%
def compute_smallest_eigenvalue_ATA(physics, im_size, max_iter=100, tol=1e-6, device="cpu"):
    """
    Computes the smallest non-zero eigenvalue of A^T*A using inverse iteration with CG.
    """

    # Start with a random vector in the image domain.
    vec = torch.randn(im_size, device=device)
    vec /= torch.linalg.norm(vec)
    
    # 1. Define the matvec function for the LinearOperator.
    # This is the same function as before, but we'll call it `matvec` for clarity.
    def AtA_matvec(x_vec: np.ndarray) -> np.ndarray:
        x_tensor = torch.from_numpy(x_vec).reshape(im_size).to(device).float()
        with torch.no_grad():
            y_tensor = physics.A_adjoint(physics.A(x_tensor))
        return y_tensor.cpu().numpy().flatten()

    # 2. THE FIX: Create a SciPy LinearOperator object.
    # This provides SciPy's CG solver with the necessary metadata (shape, dtype).
    num_pixels = np.prod(im_size)
    operator_shape = (num_pixels, num_pixels)
    # Ensure dtype matches the numpy arrays we will be using.
    operator_dtype = vec.cpu().numpy().dtype
    
    AtA_linear_op = LinearOperator(
        shape=operator_shape,
        matvec=AtA_matvec,
        dtype=operator_dtype
    )

    lambda_old = 0.0
    print("Starting inverse power iteration to find smallest eigenvalue...")

    for i in range(max_iter):
        vec_numpy = vec.cpu().numpy().flatten()

        # 3. Pass the LinearOperator object to CG, not the raw Python function.
        vec_new_numpy, info = cg(AtA_linear_op, vec_numpy, tol=1e-3, maxiter=50)
        
        if info != 0:
            print(f"Warning: CG solver did not converge at iteration {i+1}. Info: {info}")

        vec_new = torch.from_numpy(vec_new_numpy).reshape(im_size).to(device)

        lambda_new = torch.dot(vec.flatten(), vec.flatten()) / torch.dot(vec.flatten(), vec_new.flatten())
        vec = vec_new / torch.linalg.norm(vec_new)

        if abs(lambda_new - lambda_old) / (abs(lambda_new) + 1e-9) < tol:
            print(f"Converged at iteration {i+1}")
            break
        lambda_old = lambda_new

    return lambda_new.item()

# loop through some configurations

# Set up  physics model
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
im_size = (1, 1, 128, 128)
# angles = 360
#limited view angles
angles = torch.linspace(20, 160, steps=140).to(device)
#noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0/20))
physics = dinv.physics.Tomography(img_width=im_size[-1], 
                                  angles=angles, 
                                  #noise_model=noise_model,
                                  device=device)

# Call the function with physics
largest_eig = compute_largest_eigenvalue_ATA(physics, im_size, device=device)

print(f"\nUsing custom power iteration function:")
print(f"The largest eigenvalue of A^T*A is: {largest_eig:.4f}")
print(f"The Lipschitz constant of A is: {largest_eig**0.5:.4f}")

print(f"Compare with approximate operator norm of A^T A: { torch.pi / (2 * angles)}")
print(f"And approximate Lip constant of A: {1/(torch.pi / (2 * angles))}")  

# Call the function to compute smalles EV
s_eig = compute_smallest_eigenvalue_ATA(physics, im_size, device=device)

print(f"\nThe smallest non-zero eigenvalue of A^T*A is: {s_eig:.4e}")
print(f"The strong-convexity constant m of A is: {s_eig**0.5:.4f}")

# %%
