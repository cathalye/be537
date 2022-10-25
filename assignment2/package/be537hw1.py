"""Functions from the first BE537 Python assignment."""

# Import required libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import interpolate, signal, cluster
import torch
import torch.nn.functional as tfun


def my_read_nifti(filename):
    """ Read NIfTI image voxels and header 
    
    :param filename: path to the image to read
    :returns: tuple (img,hdr) consisting of numpy voxel array and nibabel NIfTI header
    """
    img = nib.load(filename)
    return img.get_fdata(), img.header


def my_write_nifti(filename, img, header = None):
    """ Write NIfTI image voxels and header 
    
    :param filename: path to the image to save
    :param img: numpy voxel array
    :param header: nibabel NIfTI header
    """
    if header is not None:
        nifti_img = nib.Nifti1Image(img, affine=header.get_best_affine(), header=header)
    else:
        nifti_img = nib.Nifti1Image(img, affine=np.eye(len(img.shape)+1))
    nib.save(nifti_img, filename)
    
    
def my_view_axis(ax, data, aspect, xhair, crange=None, cmap='gray'):
    if crange is None:
        im = ax.imshow(data, cmap=cmap, aspect=aspect)
    else:
        im = ax.imshow(data, cmap=cmap, aspect=aspect, vmin=crange[0], vmax=crange[1])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.axvline(x=xhair[0], color='lightblue')
    ax.axhline(y=xhair[1], color='lightblue')
    return im


def my_view(img, header=None, xhair=None, crange=None, cmap='gray'):
    """Display a 3D image in a layout similar to ITK-SNAP
    
    :param img: 3D voxel array
    :param header: Image header (returned by my_read_nifti)
    :param xhair: Crosshair position (1D array or tuple)
    :param crange: Intensity range, a tuple with minimum and maximum values
    :param cmap: Colormap (a string, see matplotlib documentation)
    """
    fig, axs = plt.subplots(2,2)
    xhair = np.array(img.shape)//2 if xhair is None else xhair
    crange = (np.min(img),np.max(img)) if crange is None else crange
    sp = header.get_zooms() if header is not None else np.ones((len(img.shape),1))
    im0 = my_view_axis(axs[0,0], img[:,:,xhair[2]].T, 
                       aspect=sp[1]/sp[0], xhair=(xhair[0],xhair[1]), cmap=cmap, crange=crange)
    im1 = my_view_axis(axs[0,1], img[xhair[0],:,:].T, 
                       aspect=sp[2]/sp[1], xhair=(xhair[1],xhair[2]), cmap=cmap, crange=crange)
    im2 = my_view_axis(axs[1,1], img[:,xhair[1],:].T, 
                       aspect=sp[2]/sp[0], xhair=(xhair[0],xhair[2]), cmap=cmap, crange=crange)
    axs[1,0].axis('off');
    cax = plt.axes([0.175, 0.15, 0.3, 0.05])
    plt.colorbar(im1, orientation='horizontal', ax=axs[1,1], cax=cax)
    
    
# Read the matrix from file
def my_read_transform(filename):
    """
    Read Greedy-style 3D transform (4x4 matrix) from file
    
    :param filename: File name containing transform file
    :returns: tuple (A,b) where A is the 3x3 affine matrix, b is the translation vector
    """
    M = np.loadtxt(filename)
    return (M[0:3,0:3], M[0:3,3])


# Define function to apply affine transform to an image
def my_transform_image(I_ref, I_mov, A, b, method='linear', fill_value=0):
    """
    Transform a moving image into the space of the fixed image
    
    :param I_ref: 3D voxel array of the fixed (reference) image
    :param I_mov: 3D voxel array of the moving image
    :param A: 3x3 affine transformation matrix
    :param A: 3x1 translation vector
    :param method: Interpolation method (e.g., 'linear', 'nearest')
    :param fill_value: Value with which to replace missing values (e.g., 0)   
    """
    dim_ref=I_ref.shape
    dim_mov=I_mov.shape

    rng_ref = (range(dim_ref[0]), range(dim_ref[1]), range(dim_ref[2]))
    rng_mov = (range(dim_mov[0]), range(dim_mov[1]), range(dim_mov[2]))

    # Extract pixel coordinates in fixes image domain
    px,py,pz=np.meshgrid(*rng_ref, indexing='ij')

    # Apply matrix to the coordinates, transforming them into sampling coordinates
    qx = A[0,0]*px + A[0,1]*py + A[0,2]*pz + b[0]
    qy = A[1,0]*px + A[1,1]*py + A[1,2]*pz + b[1]
    qz = A[2,0]*px + A[2,1]*py + A[2,2]*pz + b[2]

    # Interpolate moving image at sampling coordinates
    return interpolate.interpn(rng_mov, I_mov, (qx,qy,qz), 
                               bounds_error=False, fill_value=fill_value, method=method)


# Function to generate a 3D Gaussian filter
def my_gaussian_3d(sigma):
    t_range = np.arange(-np.ceil(3.5*sigma), np.ceil(3.5*sigma)+1)
    gaussian_1d = np.exp(- t_range**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    return np.tensordot(np.tensordot(gaussian_1d, gaussian_1d, axes=0),
                        gaussian_1d, axes=0)


# Function to perform FFT-based convolution with a Gaussion
def my_gaussian_lpf(image, sigma):
    gaussian_3d = my_gaussian_3d(sigma)
    return signal.fftconvolve(image, gaussian_3d, 'same')


# Function to perform mean filtering
def my_mean_lpf(image, radius):
    mean_filter = np.ones((radius*2+1,)*3)
    mean_filter /= np.sum(mean_filter)
    return signal.fftconvolve(image, mean_filter, 'same')


def my_create_pytorch_grid(img_tensor_dim, **kwargs):
    """
    Generate an identity grid for use with grid_sample, similar to meshgrid in NumPy
    
    :param img_tensor_dim: Dimensions of tensor holding the reference image (tuple of 5 values)
    :param dtype: PyTorch data type for the grid (see torch.Tensor)
    :param device: PyTorch device for the grid (see torch.Tensor)
    :returns: 5D tensor of dimension [1,S_x,S_y,S_z,3] containing the grid
    """
    
    # Generate a 3x4 representation of and identity matrix
    T_idmat = torch.eye(3,4).unsqueeze(0)

    # Update the type and device of the grid
    T_dummy = torch.zeros(1, **kwargs)
        
    # Generate a sampling grid inside of a no_grad block
    T_grid = tfun.affine_grid(T_idmat, img_tensor_dim, align_corners=False).type(T_dummy.dtype).to(T_dummy.device)
    
    return T_grid


def my_pytorch_coord_transform(img_size):
    S = np.eye(3)[(2,1,0),:]
    W = S @ (np.diag(2.0 / np.array(img_size)))
    z = S @ (1.0 / np.array(img_size) - 1)
    return W,z


def my_numpy_affine_to_pytorch_affine(A, b, img_size):
    """
    Convert affine transform (A,b) from NumPy to PyTorch coordinates
    
    :param A: affine matrix, represented as a shape (3,3) NumPy array 
    :param b: translation vector, represented as a shape (3) NumPy array
    :returns: tuple of NumPy arrays (A',b') holding affine transform in PyTorch coords
    """
    W,z = my_pytorch_coord_transform(img_size)
    A_prime = (W @ A) @ np.linalg.inv(W)
    b_prime = W @ b + z - A_prime @ z
    return A_prime, b_prime


def my_pytorch_affine_to_numpy_affine(A, b, img_size):
    """
    Convert affine transform (A,b) from PyTorch to NumPy coordinates
    
    :param A: affine matrix, represented as a shape (3,3) NumPy array
    :param b: translation vector, represented as a shape (3)NumPy array
    :returns: tuple of NumPy arrays (A',b') holding affine transform in NumPy coords
    """
    W,z = my_pytorch_coord_transform(img_size)
    W_inv = np.linalg.inv(W)
    z_inv = - W_inv @ z
    A_prime = (W_inv @ A) @ W
    b_prime = W_inv @ b + z_inv - A_prime @ z_inv
    return A_prime, b_prime


def my_apply_affine_to_pytorch_grid(A, b, grid):
    px = grid[0,:,:,:,0]
    py = grid[0,:,:,:,1]
    pz = grid[0,:,:,:,2]    
    qx = A[0,0]*px + A[0,1]*py + A[0,2]*pz + b[0]
    qy = A[1,0]*px + A[1,1]*py + A[1,2]*pz + b[1]
    qz = A[2,0]*px + A[2,1]*py + A[2,2]*pz + b[2]
    return torch.stack((qx,qy,qz),3).unsqueeze(0).type(grid.dtype)


# Function to apply the affine transform in voxel coordinates to a moving image represented
# as a PyTorch tensor
def my_transform_image_pytorch(T_ref, T_mov, T_A, T_b, 
                               mode='bilinear', padding_mode='zeros', T_grid=None):
    """
    Apply an affine transform to 3D images represented as PyTorch tensors
    
    :param T_ref: Fixed (reference) image, represented as a 5D tensor
    :param T_mov: Moving image, represented as a 5D tensor
    :param T_A: affine matrix in PyTorch coordinate space, represented as a shape (3,3) tensor
    :param T_b: translation vector in PyTorch coordinate space, represented as a shape (3) tensor
    :param mode: Interpolation mode, see grid_sample
    :param padding_mode: Padding mode, see grid_sample
    :param grid: Optional sampling grid (otherwise will call my_create_pytorch_grid) 
    :returns: Transformed moving image, represented as a 5D tensor
    """
    if T_grid is None:
        T_grid = my_create_pytorch_grid(T_ref.shape, dtype=T_ref.dtype, device=T_ref.device)
        
    T_grid_xform = my_apply_affine_to_pytorch_grid(T_A, T_b, T_grid)
    T_fu_reslice = tfun.grid_sample(T_mov, T_grid_xform, 
                                    mode=mode, padding_mode=padding_mode, 
                                    align_corners=False)
    return T_fu_reslice


def my_affine_objective_fn(T_ref, T_mov, T_A, T_b):
    """
    Compute the affine registration objective function
    
    :param T_ref: Fixed (reference) image, represented as a 5D tensor
    :param T_mov: Moving image, represented as a 5D tensor
    :param T_A: affine matrix in PyTorch coordinate space, represented as a shape (3,3) tensor
    :param T_b: translation vector in PyTorch coordinate space, represented as a shape (3) tensor
    :returns: RMS difference between the reference image and transformed moving image
    """
    T_reslice =  my_transform_image_pytorch(T_ref, T_mov, T_A, T_b)
    n_vox = np.array(T_ref.shape).prod()
    rms_diff = torch.sqrt(torch.square(T_reslice - T_ref).sum() / n_vox)
    return rms_diff