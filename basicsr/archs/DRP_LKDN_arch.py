import torch
from torch import nn as nn

from basicsr.archs.DRP_LKDN_blocks import DRP_LKDB, BSConvU, BSConvU_rep, DPR_BSConvU_g2_4, UpsampleOneStep, Upsampler_rep,DPR_BSConvU_g2_4_8
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DRP_LKDN(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=56,
                 num_atten=56,
                 num_block=8,
                 upscale=2,
                 num_in=4,
                 conv='DPR_BSConvU_g2_4',
                 upsampler='pixelshuffledirect',
                 deploy=False):
        super().__init__()
        self.num_in = num_in
        kwargs = {}
        if conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        elif conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'DPR_BSConvU_g2_4':
            self.conv = DPR_BSConvU_g2_4
            kwargs = {'deploy': deploy}
            kwargs = {'deploy': deploy}
        elif conv == 'DPR_BSConvU_g2_4_8':
            self.conv = DPR_BSConvU_g2_4_8
            kwargs = {'deploy': deploy}
        else:
            raise NotImplementedError(f'conv {conv} is not supported yet.')
        print(conv)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B2 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B3 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B4 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B5 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B6 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B7 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)
        self.B8 = DRP_LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv,**kwargs)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle_rep':
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        input = torch.cat([input] * self.num_in, dim=1)
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
