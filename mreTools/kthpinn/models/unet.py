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

        self.p_dropout = 0.01

        # input: n x m x 3
        self.e11 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.bnd1 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bnd2 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bnd3 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bnd4 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bnd5 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.up1 = nn.UpsamplingBilinear2d(self.image_size_e4)
        self.up1_ = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.do1 = nn.Dropout2d(self.p_dropout)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bnu1 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up2 = nn.UpsamplingBilinear2d(self.image_size_e3)
        self.up2_ = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.do2 = nn.Dropout2d(self.p_dropout)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bnu2 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up3 = nn.UpsamplingBilinear2d(self.image_size_e2)
        self.up3_ = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.do3 = nn.Dropout2d(self.p_dropout)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bnu3 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up4 = nn.UpsamplingBilinear2d(self.input_image_size)
        self.up4_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.bnu4 = nn.BatchNorm2d(64)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.do4 = nn.Dropout2d(self.p_dropout)
        self.d42 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        
        xe11 = self.activation(self.e11(x))
        be1 = self.bnd1(xe11)
        xe12 = self.activation(self.e12(be1))
        xp1 = self.pool1(xe12)

        xe21 = self.activation(self.e21(xp1))
        be2 = self.bnd2(xe21)
        xe22 = self.activation(self.e22(be2))
        xp2 = self.pool2(xe22)

        xe31 = self.activation(self.e31(xp2))
        be3 = self.bnd3(xe31)
        xe32 = self.activation(self.e32(be3))
        xp3 = self.pool3(xe32)

        xe41 = self.activation(self.e41(xp3))
        be4 = self.bnd4(xe41)
        xe42 = self.activation(self.e42(be4))
        # xp4 = self.pool4(xe42)

        # xe51 = self.activation(self.e51(xp4))
        # be5 = self.bnd5(xe51)
        # xe52 = self.activation(self.e52(be5))
        
        # # Decoder
        # xu1 = self.up1_(self.up1(xe52))
        # xu11 = torch.cat([xu1, xe42], dim=1) # skip connection
        # xd11 = self.activation(self.d11(xu11))
        # bd1 = self.bnu1(xd11)
        # xd12 = self.activation(self.d12(bd1))

        xu2 = self.up2_(self.up2(xe42))
        # xu2 = self.up2_(self.up2(xd12))
        xu22 = torch.cat([xu2, xe32], dim=1) # skip connection
        xu22 = self.do2(xu22)
        xd21 = self.activation(self.d21(xu22))
        bd2 = self.bnu2(xd21)
        xd22 = self.activation(self.d22(bd2))

        xu3 = self.up3_(self.up3(xd22))
        xu33 = torch.cat([xu3, xe22], dim=1) # skip connection
        xu33 = self.do3(xu33)
        xd31 = self.activation(self.d31(xu33))
        bd3 = self.bnu3(xd31)
        xd32 = self.activation(self.d32(bd3))

        xu4 = self.up4_(self.up4(xd32))
        xu44 = torch.cat([xu4, xe12], dim=1) # skip connection
        xd41 = self.activation(self.d41(xu44))
        xd41 = self.do4(xd41)
        bd4 = self.bnu4(xd41)
        xd42 = self.d42(bd4)

        out = xd42

        return out