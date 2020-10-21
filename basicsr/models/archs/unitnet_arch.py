import functools
import torch
import torch.nn as nn
from torch.nn import init as init
import torch.nn.functional as F
import numpy as np

# import pdb
import math


class UnitNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ns, kpn_sz=5, skip=False, norm_type=None, \
                    kernel_cond=None, act_type='leakyrelu', \
                    res_scale=1, kernel_size=3):
        super().__init__()
        self.ns = ns
        self.skip = skip
        self.kpn_sz = kpn_sz
        self.kernel_cond = kernel_cond

        #############################
        # Restoration Branch
        #############################
        # Encoder
        self.conv_11 = conv_block(in_nc, nf, kernel_size=kernel_size, act_type=act_type)
        self.conv_12 = ResBlock(nf, res_scale=res_scale, act_type=act_type)
        self.conv_13 = ResBlock(nf, res_scale=res_scale, act_type=act_type)

        self.conv_21 = conv_block(nf, 2*nf, stride=2, kernel_size=kernel_size, act_type=act_type)
        self.conv_22 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)
        self.conv_23 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)

        self.conv_31 = conv_block(2*nf, 4*nf, stride=2, kernel_size=kernel_size, act_type=act_type)
        self.conv_32 = ResBlock(4*nf, res_scale=res_scale, act_type=act_type)
        self.conv_33 = ResBlock(4*nf, res_scale=res_scale, act_type=act_type)

        # Decoder
        self.upconv_21 = upconv(4*nf, 2*nf, 2, act_type=act_type)
        self.upconv_22 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)
        self.upconv_23 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)

        self.upconv_11 = upconv(2*nf, nf, 2, act_type=act_type)
        self.upconv_12 = ResBlock(nf, res_scale=res_scale, act_type=act_type)
        self.upconv_13 = ResBlock(nf, res_scale=res_scale, act_type=act_type)

        self.final_conv = conv_block(nf, out_nc, kernel_size=kernel_size, act_type=act_type)

        #############################
        # Kernel Prediction Branch
        #############################
        if self.kernel_cond:
            if self.kernel_cond == 'img':
                cond_nc = in_nc
            elif self.kernel_cond == 'psf':
                cond_nc = 7
                self.register_buffer('psf', get_pca())
            elif self.kernel_cond == 'img-psf':
                cond_nc = in_nc + 7
                self.register_buffer('psf', get_pca())

            self.kconv_11 = conv_block(cond_nc, nf, kernel_size=kernel_size, act_type=act_type)
            self.kconv_12 = ResBlock(nf, res_scale=res_scale, act_type=act_type)
            self.kconv_13 = ResBlock(nf, res_scale=res_scale, act_type=act_type)

            self.kconv_21 = conv_block(nf, 2*nf, stride=2, kernel_size=kernel_size, act_type=act_type)
            self.kconv_22 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)
            self.kconv_23 = ResBlock(2*nf, res_scale=res_scale, act_type=act_type)

            self.kconv_31 = conv_block(2*nf, 4*nf, stride=2, kernel_size=kernel_size, act_type=act_type)
            self.kconv_32 = ResBlock(4*nf, res_scale=res_scale, act_type=act_type)
            self.kconv_33 = ResBlock(4*nf, res_scale=res_scale, act_type=act_type)

            self.dynamic_kernel = nn.Sequential(
                conv_block(4*nf, 4*nf, kernel_size=kernel_size),
                ResBlock(4*nf, res_scale=res_scale, act_type=act_type),
                ResBlock(4*nf, res_scale=res_scale, act_type=act_type),
                conv_block(4*nf, 4*nf * (kpn_sz ** 2), kernel_size=1))


    def forward(self, x):
        if not self.training:
            N, C, H, W = x.shape
            H_pad = 4 - H % 4 if not H % 4 == 0 else 0
            W_pad = 4 - W % 4 if not W % 4 == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        #############################
        # Kernel Prediction Branch
        #############################
        # kernel network
        if self.kernel_cond:
            if self.kernel_cond == 'img':
                cond_x = x
            elif self.kernel_cond == 'psf':
                cond_x = self.psf.expand(x.shape[0], -1, x.shape[2], x.shape[3])
            elif self.kernel_cond == 'img-psf':
                cond_x = self.psf.expand(x.shape[0], -1, x.shape[2], x.shape[3])
                cond_x = torch.cat((cond_x, x), dim=1)

            kfea1 = self.kconv_13(self.kconv_12(self.kconv_11(cond_x)))
            kfea2 = self.kconv_23(self.kconv_22(self.kconv_21(kfea1)))
            kfea3 = self.kconv_33(self.kconv_32(self.kconv_31(kfea2)))
            dynamic_kernel = self.dynamic_kernel(kfea3)


        #############################
        # Restoration Branch
        #############################
        # Encoder
        fea1 = self.conv_13(self.conv_12(self.conv_11(x)))
        fea2 = self.conv_23(self.conv_22(self.conv_21(fea1)))
        fea3 = self.conv_33(self.conv_32(self.conv_31(fea2)))

        # Dynamic convolution
        if self.kernel_cond:
            fea3 = kernel2d_conv(fea3, dynamic_kernel, self.kpn_sz)

        # Decoder
        if self.skip:
            upfea2 = self.upconv_23(self.upconv_22(self.upconv_21(fea3) + fea2))
        else:
            upfea2 = self.upconv_23(self.upconv_22(self.upconv_21(fea3)))

        if self.skip:
            upfea1 = self.upconv_13(self.upconv_12(self.upconv_11(upfea2) + fea1))
        else:
            upfea1 = self.upconv_13(self.upconv_12(self.upconv_11(upfea2)))

        fea = self.final_conv(upfea1)
        out = fea + x

        if not self.training:
            out = out[:, :, :H, :W]
        return out


def get_pca():
    x = torch.tensor([0.2038666, -0.30429688, -0.25263874, -0.07093838, 0.00750307, \
        0.00584931, -0.03297379])
    return x[None, :, None, None]


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def upconv(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


    """
    Cascade Channel Attention Block, 3-3 style
    """

    def __init__(self, nc, gc, kernel_size=3, stride=1, dilation=1, groups=1, reduction=16, \
            bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(CCAB, self).__init__()
        self.nc = nc
        self.RCAB = nn.ModuleList([RCAB(gc, kernel_size, reduction, stride, dilation, groups, bias, pad_type, \
                    norm_type, act_type, mode, res_scale) for _ in range(nc)])
        self.CatBlocks = nn.ModuleList([conv_block((i + 2)*gc, gc, kernel_size=1, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nc)])

    def forward(self, x):
        pre_fea = x
        for i in range(self.nc):
            res = self.RCAB[i](x)
            pre_fea = torch.cat((pre_fea, res), dim=1)
            x = self.CatBlocks[i](pre_fea)
        return x

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResBlock(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, act_type='leakyrelu'):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.act = act(act_type) if act_type else None

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.act(self.conv1(x)))
        return identity + out * self.res_scale


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer, 
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, axis=-1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out
