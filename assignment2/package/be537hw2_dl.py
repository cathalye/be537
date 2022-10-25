"""Helpful functoins for the deep learning portion of the second assignment"""

import sys
import glob
import torch.utils.data
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as tfun
from torchvision import transforms

# Import the functions from the first assignment
from be537hw1 import *
from be537hw2 import *


class AssignmentTwoDataset(torch.utils.data.Dataset):
    """PyTorch DataSet for loading registration pairs from the data directory"""
    
    def __init__(self, file_pattern='atlas_*.nii.gz', downsample=2.0, transform=None, **kwargs):
        
        # Store parameters
        self.file_pattern = file_pattern
        self.downsample = downsample
        self.transform = transform
        
        # List images
        self.nii_fn = glob.glob(file_pattern)
        
        # Downsample images and cache (possibly in GPU memory)
        self.nii = []
        for fn in glob.glob(file_pattern):
            (I, hdr) = my_read_pytorch_image_from_nifti(fn, **kwargs)
            I_ds = my_image_downsample(I, downsample)
            hdr_ds = my_adjust_nifti_header_for_resample(hdr, I_ds.shape[2:5])
            self.nii.append((I_ds, hdr_ds))
        
        # Create a list of registration pairs
        self.pair_idx = []
        for i,_ in enumerate(self.nii):
            for j,_ in enumerate(self.nii):
                if i != j:
                    self.pair_idx.append((i,j))
        
    def __len__(self):
        return len(self.pair_idx)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the pair of image indices
        (j_fix,j_mov) = self.pair_idx[idx]
        
        # Read the fixed and moving images
        I_fix, hdr_fix = self.nii[j_fix]
        I_mov, hdr_mov = self.nii[j_mov]
        
        # Stack the images in the channel dimensions
        I = torch.cat((I_fix, I_mov), dim=1)
        
        sample = { 'image': I.squeeze(), 'hdr': hdr_fix }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def my_random_rotation(sigma=0.1745):
    
    # Generate a random rotation axis 
    k = (lambda x : x / np.linalg.norm(x))(np.random.randn(3))
    K = np.reshape([0,-k[2],k[1],k[2],0,-k[0],-k[1],k[0],0],(3,3))
    
    # Generate a random rotation angle
    theta = np.random.randn(1) * sigma
    
    # Organize K into a matrix
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def my_random_affine(sigma_rot=0.1745, sigma_skew=0.1745, sigma_log_scale=0.1, sigma_shift=0.05):
    R = my_random_rotation(sigma_rot)
    B = my_random_rotation(sigma_skew)
    S = np.diag(np.exp(np.random.randn(3)*sigma_log_scale))
    b = np.random.randn(3)*sigma_shift
    return R @ (B.T @ S @ B), b
    
    
class RandomAffineImage3D(object):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __call__(self, sample):
        A,b = my_random_affine(**self.kwargs)
        I = sample['image'].unsqueeze(0)
        J1 = my_transform_image_pytorch(I[:,0:1,:,:,:], I[:,0:1,:,:,:], torch.tensor(A), torch.tensor(b))
        J2 = my_transform_image_pytorch(I[:,1:2,:,:,:], I[:,1:2,:,:,:], torch.tensor(A), torch.tensor(b))
        return { 'image': torch.cat((J1,J2), dim=1).squeeze(), 'hdr': sample['hdr'] }
    
    
class PowerOfTwoCenterCrop(object):
    
    def __init__(self, exponent=3):
        self.factor = 2 ** exponent
        
    def __call__(self, sample):
        I = sample['image']
        dim = np.array(I.shape[1:4])
        dim_new = np.floor(dim / self.factor) * self.factor
        a = np.int32(np.floor(0.5 * (dim - dim_new)))
        b = a + np.int32(dim_new)
        I_crop = I[:,a[0]:b[0],a[1]:b[1],a[2]:b[2]]
        return { 'image': I_crop, 'hdr': sample['hdr'] }  
    
    
class PowerOfTwoCenterPad(object):
    
    def __init__(self, exponent=3):
        self.factor = 2 ** exponent
        
    def __call__(self, sample):
        I = sample['image']
        dim = np.array(I.shape[1:4])
        dim_new = np.ceil(dim / self.factor) * self.factor
        a = np.int32(np.floor(0.5 * (dim_new - dim)))
        b = np.int32(dim_new - dim) - a
        I_pad = tfun.pad(I, (a[2],b[2],a[1],b[1],a[0],b[0]))
        return { 'image': I_pad, 'hdr': sample['hdr'] }  
    
    
class NormalizeIntensity(object):
    
    def __init__(self, percentile=0.99):
        self.percentile = percentile
        
    def __call__(self, sample):
        
        I = sample['image']
        val = torch.quantile(I, self.percentile)
        return {'image': torch.clamp(I / val, max=1.0), 'hdr': sample['hdr'] }
    
    
def my_collate_fn(batch):
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'hdr': [x['hdr'] for x in batch] }


# ====================================================================================
# Convolutional VAE implementation, adapted from https://github.com/LukeDitria/CNN-VAE
# ====================================================================================

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(torch.nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_down, self).__init__()
        
        self.conv1 = torch.nn.Conv3d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = torch.nn.BatchNorm3d(channel_out//2)
        self.conv2 = torch.nn.Conv3d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = torch.nn.BatchNorm3d(channel_out)
        
        self.conv3 = torch.nn.Conv3d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = torch.nn.AvgPool3d(scale,scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = tfun.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = tfun.rrelu(x + skip)
        return x

    
#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(torch.nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()
        
        self.conv1 = torch.nn.Conv3d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = torch.nn.BatchNorm3d(channel_out//2)
        self.conv2 = torch.nn.Conv3d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = torch.nn.BatchNorm3d(channel_out)
        
        self.conv3 = torch.nn.Conv3d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = torch.nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = tfun.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = tfun.rrelu(x + skip)
        return x
    

#Encoder block
#Built for a 64x64x3 image and will result in a latent vector of size Z x 1 x 1 
#As the network is fully convolutional it will work for other larger images sized 2^n the latent
#feature map size will just no longer be 1 - aka Z x H x W
class Encoder(torch.nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)#64
        self.conv2 = Res_down(ch, 2*ch)#32
        self.conv3 = Res_down(2*ch, 4*ch)#16
        self.conv4 = Res_down(4*ch, 8*ch)#8
        self.conv_mu = torch.nn.Conv3d(8*ch, z, 2, 2)#2
        self.conv_logvar = torch.nn.Conv3d(8*ch, z, 2, 2)#2

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, Train = True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if Train:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return x, mu, logvar
    
#Decoder block
#Built to be a mirror of the encoder block
class Decoder(torch.nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch*8)
        self.conv3 = Res_up(ch*8, ch*4)
        self.conv4 = Res_up(ch*4, ch*2)
        self.conv5 = Res_up(ch*2, ch)
        self.conv6 = Res_up(ch, ch//2)
        self.conv7 = torch.nn.Conv3d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 
    
#VAE network, uses the above encoder and decoder blocks 
class VAE(torch.nn.Module):
    def __init__(self, channel_in, channel_out, ch = 64, z = 512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""
        
        self.encoder = Encoder(channel_in, ch = ch, z = z)
        self.decoder = Decoder(channel_out, ch = ch, z = z)

    def forward(self, x, Train = True):
        encoding, mu, logvar = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon, mu, logvar