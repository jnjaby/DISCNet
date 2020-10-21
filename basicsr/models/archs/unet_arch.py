import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# import models.archs.arch_util as arch_util

# import pdb
import math


class UNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nc, nb, reduction, groups=1, norm_type=None, act_type='leakyrelu', \
            mode='NAC', inner_arch='OPN', res_scale=1, kernel_size=3, 
            use_guidance=False, use_attention=False):
        super(UNet, self).__init__()
        self.nb = nb
        self.inner_arch = inner_arch
        self.use_guidance = use_guidance
        self.use_attention = use_attention

        if self.use_attention:
            assert self.use_guidance is True, (
                'There should be guidance map when using attention')

        if self.use_guidance:
            self.conv_11 = conv_block(in_nc + 1, nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        else:
            self.conv_11 = conv_block(in_nc, nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv_12 = conv_block(nf, nf, stride=2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)

        self.conv_21 = conv_block(nf, 2*nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv_22 = conv_block(2*nf, 2*nf, stride=2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)

        self.conv_41 = conv_block(2*nf, 4*nf, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)

        if inner_arch == 'OPN':
            self.CascadeBlocks = nn.ModuleList([CCAB(nc, 4*nf, kernel_size=3, groups=groups, reduction=16, \
                norm_type=norm_type, act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)])
            self.CatBlocks = nn.ModuleList([conv_block((i + 2)*4*nf, 4*nf, kernel_size=1, \
                norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nb)])
        elif inner_arch == 'RRDB':
            RRDBs = [RRDB(4*nf, gc=nf, norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nb)]
            self.model = sequential(*RRDBs)
        elif inner_arch == 'ResNet':
            resnet_blocks = [ResNetBlock(4*nf, 4*nf, 4*nf, norm_type=norm_type, act_type=act_type,\
                mode=mode, res_scale=res_scale) for _ in range(nb)]
            self.model = sequential(*resnet_blocks)
        else:
            raise NotImplementedError('Inner architecture [{:s}] is not recognized.'.format(inner_arch))

        self.conv_43 = pixelshuffle_block(4*nf, 2*nf, 2, act_type=act_type)

        self.conv_23 = conv_block(4*nf, 2*nf, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv_24 = pixelshuffle_block(2*nf, nf, 2, act_type=act_type)

        self.conv_13 = conv_block(2*nf, nf, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv_14 = conv_block(nf, out_nc, kernel_size=kernel_size, norm_type=None, act_type=None)


    def generate_map(self, x):
        # coef = torch.tensor((0.299, 0.587, 0.114)).to(x.device)
        # attention_map = x * coef[None, ..., None, None]
        # attention_map = attention_map.sum(axis=1, keepdims=True)
        attention_map = x[:, [-1], :, :]
        return attention_map


    def forward(self, x):
        if not self.training:
            N, C, H, W = x.shape
            H_pad = 4 - H % 4 if not H % 4 == 0 else 0
            W_pad = 4 - W % 4 if not W % 4 == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        if self.use_attention:
            attention_map = self.generate_map(x)
            map_down1 = F.avg_pool2d(attention_map, kernel_size=2, stride=2)
            # print('generate attention map ...')
            # map_down2 = F.avg_pool2d(map_down1, kernel_size=2, stride=2)

            # x = torch.cat((x, attention_map), dim=1)

        x = self.conv_11(x)
        if self.use_attention:
            fea1 = x * attention_map
        else:
            fea1 = x

        x = self.conv_12(x)
        x = self.conv_21(x)
        if self.use_attention:
            fea2 = x * map_down1
        else:
            fea2 = x

        x = self.conv_22(x)
        x = self.conv_41(x)

        if self.inner_arch == 'OPN':
            pre_fea = x
            for i in range(self.nb):
                res = self.CascadeBlocks[i](x)
                pre_fea = torch.cat((pre_fea, res), dim=1)
                x = self.CatBlocks[i](pre_fea)
        else:
            x = self.model(x)

        x = self.conv_43(x)
        x = self.conv_23(torch.cat((fea2, x), dim=1))
        x = self.conv_24(x)
        x = self.conv_13(torch.cat((fea1, x), dim=1))
        x = self.conv_14(x)

        if not self.training:
            x = x[:, :, :H, :W]
        return x



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

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
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


class CALayer(nn.Module):
    # Channel Attention (CA) Layer
    def __init__(self, channel, reduction=16, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.attention = sequential(
                nn.AdaptiveAvgPool2d(1),
                conv_block(channel, channel // reduction, 1, stride, dilation, groups, bias, pad_type, \
                            norm_type, act_type, mode),
                conv_block(channel // reduction, channel, 1, stride, dilation, groups, bias, pad_type, \
                            norm_type, None, mode),
                nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class RCAB(nn.Module):
    ## Residual Channel Attention Block (RCAB)
    def __init__(self, nf, kernel_size=3, reduction=16, stride=1, dilation=1, groups=1, bias=True, \
            pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(RCAB, self).__init__()
        self.res = sequential(
            conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, pad_type, \
                        norm_type, act_type, mode),
            conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, pad_type, \
                        norm_type, None, mode),
            CALayer(nf, reduction, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class CCAB(nn.Module):
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