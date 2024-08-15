from mreTools.kthpinn.models.unet import UNet
import torch
from torch import nn

class KthMreUNet(nn.Module):
    """Implementation of El-Unet introduced in Kamali & Laksari (2023)
    """
    def __init__(self, image_size):
        super().__init__()

        self.input_image_size = image_size

        """unet_u
        The task of this first network is to learn a continuous representation of the displacement field 
        in order to get pytorch gradients.
        """
        self.unet_u = UNet(image_size)
        """unet_mu
        The task of this second network is to learn the complex shear modulus from the continuous displacement returned from unet_u."""
        self.unet_mu = UNet(image_size)

    def forward(self, x):
        # input: 2 channels containing real and imaginary parts of the displacement field
        # output: 2 channels containing real and imaginary parts of the continuous displacement field
        u = self.unet_u(x)
        # input: 2 channels containing real and imaginary parts of the continuous displacement field
        # output: 2 channels containing real and imaginary parts of the complex shear modulus
        mu = self.unet_mu(u)

        return u, mu