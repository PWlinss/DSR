'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from pth.convert_test import process_grouped_params
import argparse



class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out
# BSConv添加双分支
class DPR_BSConvU_g2_4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None,deploy=False,ESDB_mode=False,num_group1=4,num_group2=2):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

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

    def forward(self, x):
        if hasattr(self, 'bs_reparam'):
            fea = self.bs_reparam(x)
            fea = self.dw(fea)
            return fea
        else:
            fea1 = self.pw(x)
            fea2 = self.ce1(x)
            fea3 = self.ce2(x)
            fea=fea1+fea2+fea3
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
#BSConv添加三分支
class DPR_BSConvU_g2_4_8(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None,deploy=False,ESDB_mode=False,num_group1=8,num_group2=4,num_group3=2):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

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

    def forward(self, x):
        if hasattr(self, 'bs_reparam'):
            fea = self.bs_reparam(x)
            fea = self.dw(fea)
            return fea
        else:
            fea1 = self.pw(x)
            fea2 = self.ce1(x)
            fea3 = self.ce2(x)
            fea4 = self.ce3(x)
            fea=fea1+fea2+fea3+fea4
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

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
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
        self.dw = torch.nn.Conv2d(
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


class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        assert 0.0 <= p <= 1.0
        # 用两次pointwise 1.当in_channels>16，mid_channels=p*in_channels
        #                2.当4<in_channels<16，mid_channels=4
        #                3.当in_channels<4,mid_channels=in_channels


        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # depthwise
        self.dw = torch.nn.Conv2d(
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

    def forward(self, x):
        fea = self.pw1(x)
        fea = self.pw2(fea)
        fea = self.dw(fea)
        return fea

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25,deploy=False):
        super(ESA, self).__init__()
        f = num_feat // 4
        kwargs = {}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}
        elif conv.__name__ == 'MyBSConvU':
            kwargs = {'deploy': deploy}
            conv = MyBSConvU
        elif conv.__name__ == 'MyBSConvU2':
            kwargs = {'deploy': deploy}
            conv = MyBSConvU2
        elif conv.__name__ == 'MyBSConvU3':
            kwargs = {'deploy': deploy}
            conv = MyBSConvU3

        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **kwargs)
        self.conv2 = conv(f, f, 3, 2, 0, **kwargs)
        self.conv3 = conv(f, f, kernel_size=3, **kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m


class DRP_ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25,deploy=False,ESDB_mode=False):
        super(DRP_ESDB, self).__init__()
        # kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}
        elif conv.__name__ == 'DPR_BSConvU_g2_4':
            kwargs = {'deploy': deploy}
            conv = DPR_BSConvU_g2_4
        elif conv.__name__ == 'DPR_BSConvU_g2_4_8':
            kwargs = {'deploy': deploy}
            conv = DPR_BSConvU_g2_4_8
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, conv,**kwargs)
        self.cca = CCALayer(in_channels)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class DRP_BSRN(nn.Module):
    def __init__(self,num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25,deploy_mode=False,ESDB_mode=False):
        super(DRP_BSRN, self).__init__()
        print("deploy_mode:{}".format(deploy_mode))
        if conv == 'BSConvS':
            kwargs = {'p': p}
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
            fea_conv = DepthWiseConv
        elif conv == 'DPR_BSConvU_g2_4':
            self.conv = DPR_BSConvU_g2_4
            kwargs = {'deploy': deploy_mode,
                      'ESDB_mode':ESDB_mode}
            fea_conv = DPR_BSConvU_g2_4
        elif conv == 'BSConvS':
            self.conv = BSConvS
            fea_conv = BSConvS
        elif conv== 'DPR_BSConvU_g2_4_8':
            self.conv = DPR_BSConvU_g2_4_8
            fea_conv = DPR_BSConvU_g2_4_8
            kwargs = {'deploy': deploy_mode,
                      'ESDB_mode':ESDB_mode}
        else:
            self.conv = nn.Conv2d
            fea_conv = nn.Conv2d
        print(conv)


        # deploy_mode = opt['deploy_mode']
        # deploy_mode = True
        # if self.conv.__name__ == 'MyBSConvU' or self.conv.__name__ == 'MyBSConvU2':
        #     kwargs = {'deploy': deploy_mode}
        self.fea_conv = fea_conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)
        self.B1 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B2 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B3 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B4 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B5 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B6 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B7 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)
        self.B8 = DRP_ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p,**kwargs)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3,**kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

