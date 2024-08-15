import torch
from torch import nn

class UNet(nn.Module):
    """Implementation of UNet inspired by Kamali & Laksari (2023). 
    Instead of strain map, we use magnitude and phase maps as input.
    Same is true for the output, since we want to predict complex images.

    While the original implementation uses ReLU activation, here other functions can be specified as well. Furthermore
    i omitted the last activation function to allow negative values in the output.
    """
    def __init__(self, image_size, activation = nn.functional.relu):
        super().__init__()
        
        self.activation = activation
        self.input_image_size = image_size
        self.image_size_e2 = ((image_size[0]) // 2, (image_size[1]) // 2)
        self.image_size_e3 = ((self.image_size_e2[0]) // 2, (self.image_size_e2[1]) // 2)
        self.image_size_e4 = ((self.image_size_e3[0]) // 2, (self.image_size_e3[1]) // 2)
        self.image_size_e5 = ((self.image_size_e4[0]) // 2, (self.image_size_e4[1]) // 2)

        # input: n x m x 3
        self.e11 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.up1 = nn.UpsamplingBilinear2d(self.image_size_e4)
        self.up1_ = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up2 = nn.UpsamplingBilinear2d(self.image_size_e3)
        self.up2_ = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up3 = nn.UpsamplingBilinear2d(self.image_size_e2)
        self.up3_ = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up4 = nn.UpsamplingBilinear2d(self.input_image_size)
        self.up4_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        
        xe11 = self.activation(self.e11(x))
        xe12 = self.activation(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.activation(self.e21(xp1))
        xe22 = self.activation(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.activation(self.e31(xp2))
        xe32 = self.activation(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.activation(self.e41(xp3))
        xe42 = self.activation(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.activation(self.e51(xp4))
        xe52 = self.activation(self.e52(xe51))
        
        # Decoder
        xu1 = self.up1_(self.up1(xe52))
        xu11 = torch.cat([xu1, xe42], dim=1) # skip connection
        xd11 = self.activation(self.d11(xu11))
        xd12 = self.activation(self.d12(xd11))

        xu2 = self.up2_(self.up2(xd12))
        xu22 = torch.cat([xu2, xe32], dim=1) # skip connection
        xd21 = self.activation(self.d21(xu22))
        xd22 = self.activation(self.d22(xd21))

        xu3 = self.up3_(self.up3(xd22))
        xu33 = torch.cat([xu3, xe22], dim=1) # skip connection
        xd31 = self.activation(self.d31(xu33))
        xd32 = self.activation(self.d32(xd31))

        xu4 = self.up4_(self.up4(xd32))
        xu44 = torch.cat([xu4, xe12], dim=1) # skip connection
        xd41 = self.activation(self.d41(xu44))
        xd42 = self.d42(xd41)

        out = xd42

        return out