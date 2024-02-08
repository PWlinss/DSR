import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from pth.convert_test import process_grouped_params


class BSConvU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class BSConvU_rep(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.rep1x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea) + fea
        fea = self.dw(fea) + fea + self.rep1x1(fea)
        return fea

class DPR_BSConvU_g2_4(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros",deploy=False,
                 num_group1=4,num_group2=2):
        super().__init__()

        if deploy:
            self.bs_reparam = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                stride=1, padding=0,dilation=1,groups=1,bias=False,)
        else:
            # pointwise
            self.pw=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),stride=1,
                    padding=0,dilation=1,groups=1,bias=False,)
            # channel enhance
            self.ce1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                    stride=1, padding=0, dilation=1,groups=num_group1,bias=False,)
            self.ce2 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                    stride=1, padding=0, dilation=1,groups=num_group2,bias=False,)
        # depthwise
        self.dw = torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,
                stride=stride,padding=padding,dilation=dilation,groups=out_channels,bias=bias,padding_mode=padding_mode,
        )
        self.rep1x1 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1, 1),
            stride=1,padding=0,dilation=dilation,groups=out_channels,bias=bias,padding_mode=padding_mode,
        )

    def forward(self, fea):
        if hasattr(self, 'bs_reparam'):
            fea = self.bs_reparam(fea)+fea
            fea = self.dw(fea)
            return fea
        else:
            fea = self.pw(fea) + self.ce1(fea) + self.ce2(fea) + fea
            fea = self.dw(fea)
            return fea
        
    def switch_to_deploy(self):
        if hasattr(self, 'bs_reparam'):
            return
        kernel1 = self.pw.weight.data
        kernel2 = self.ce1.weight.data
        kernel2 = process_grouped_params(kernel2,self.ce1.groups)
        kernel3 = self.ce2.weight.data
        kernel3 = process_grouped_params(kernel3,self.ce2.groups)
        kernel_reraram =  kernel1+kernel2+kernel3
        self.bs_reparam = nn.Conv2d(in_channels=self.pw.in_channels, out_channels=self.pw.out_channels,
                                    kernel_size=1, stride=1,padding=0, dilation=1, groups=self.pw.groups, bias=False)
        self.bs_reparam.weight.data = kernel_reraram
        self.__delattr__('pw')
        self.__delattr__('ce1')
        self.__delattr__('ce2')
        self.deploy = True

class DPR_BSConvU_g2_4_8(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros",deploy=False,
                 num_group1=2,num_group2=4,num_group3=8):
        super().__init__()

        if deploy:
            self.bs_reparam = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                stride=1, padding=0,dilation=1,groups=1,bias=False,)
        else:
            # pointwise
            self.pw=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),stride=1,
                    padding=0,dilation=1,groups=1,bias=False,)
            # channel enhance
            self.ce1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                    stride=1, padding=0, dilation=1,groups=num_group1,bias=False,)
            self.ce2 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                    stride=1, padding=0, dilation=1,groups=num_group2,bias=False,)
            self.ce3 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                    stride=1, padding=0, dilation=1,groups=num_group3,bias=False,)
        # depthwise
        self.dw = torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,
                stride=stride,padding=padding,dilation=dilation,groups=out_channels,bias=bias,padding_mode=padding_mode,
        )
        self.rep1x1 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1, 1),
            stride=1,padding=0,dilation=dilation,groups=out_channels,bias=bias,padding_mode=padding_mode,
        )

    def forward(self, fea):
        if hasattr(self, 'bs_reparam'):
            fea = self.bs_reparam(fea)
            fea = self.dw(fea)
            return fea
        else:
            fea = self.pw(fea) + self.ce1(fea) + self.ce2(fea) + self.ce3(fea)
            fea = self.dw(fea)
            return fea
        
    def switch_to_deploy(self):
        if hasattr(self, 'bs_reparam'):
            return
        kernel1 = self.pw.weight.data
        kernel2 = self.ce1.weight.data
        kernel2 = process_grouped_params(kernel2,self.ce1.groups)
        kernel3 = self.ce2.weight.data
        kernel3 = process_grouped_params(kernel3,self.ce2.groups)
        kernel4 = self.ce3.weight.data
        kernel4 = process_grouped_params(kernel4,self.ce3.groups)
        kernel_reraram =  kernel1+kernel2+kernel3+kernel4
        self.bs_reparam = nn.Conv2d(in_channels=self.pw.in_channels, out_channels=self.pw.out_channels,
                                    kernel_size=1, stride=1,padding=0, dilation=1, groups=self.pw.groups, bias=False)
        self.bs_reparam.weight.data = kernel_reraram
        self.__delattr__('pw')
        self.__delattr__('ce1')
        self.__delattr__('ce2')
        self.__delattr__('ce3')
        self.deploy = True

class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn


class DRP_LKDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d,deploy=False):
        super().__init__()

        kwargs = {}
        if conv.__name__ == 'DPR_BSConvU_g2_4':
            kwargs = {'deploy': deploy}
        elif conv.__name__ == 'DPR_BSConvU_g2_4_8':
            kwargs = {'deploy': deploy}
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if (atten_channels is None):
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, padding=1,**kwargs)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = conv(self.rc, self.rc, kernel_size=3, padding=1,**kwargs)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = conv(self.rc, self.rc, kernel_size=3, padding=1,**kwargs)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        # self.c5 = Group_Conv(self.dc * 4, self.atten_channels, 1,**kwargs)
        # self.atten = Attention(self.atten_channels)
        # self.c6 = Group_Conv(self.atten_channels, out_channels, 1,**kwargs)
        
        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)
        self.atten = Attention(self.atten_channels)
        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.atten(out)
        out_fused = self.c6(out_fused)
        out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_fused = self.pixel_norm(out_fused)
        out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out_fused + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)
