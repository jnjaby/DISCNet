#### Standard Library Imports

#### Library imports
import torch
from utils import torch_fft
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


def TorchComplexExp(phase):
    """
    e^(i*phase) =
        cos phase + i sin phase
    :param phase:
    :return: Complex output
    """
    real = torch.cos(phase)
    imag = torch.sin(phase)

    return torch.stack((real, imag), dim=-1)


def TorchComplexMul( v1_complex, v2_complex ):
    ( v1_real, v1_imag ) = v1_complex.chunk(2, dim=-1)
    ( v2_real, v2_imag ) = v2_complex.chunk(2, dim=-1)
    # Do not store intermediate real and imag arrays because they occupy A LOT of GPU memory.
    # t_real = (v1_real * v2_real) - (v1_imag * v2_imag)
    # t_imag = (v1_real * v2_imag) + (v1_imag * v2_real)
    # return torch.cat( ( t_real, t_imag  ), dim = -1 )
    return torch.cat( ( (v1_real * v2_real) - (v1_imag * v2_imag), (v1_real * v2_imag) + (v1_imag * v2_real)  ), dim = -1 )

def TorchRoll( x, n, roll_dim = -1):

    dim_size = x.shape[roll_dim]
    assert( n < dim_size ), "Error TorchRoll: n should be less than the size of dim {}".format( roll_dim )

    if( n < 0 ):
        abs_n
        (x1, x2) = torch.split( x, split_size_or_sections = ( abs(n), dim_size + n ), dim = roll_dim)
    elif( n == 0 ):
        return x
    else:
        (x1, x2) = torch.split( x, split_size_or_sections = ( dim_size-n, n ), dim = roll_dim)

    return torch.cat( (x2, x1), dim = roll_dim)

def bucketize(tensor, bucket_boundaries):
    delta_d = bucket_boundaries[1] - bucket_boundaries[0]
    n_negatives = torch.sum( bucket_boundaries < 0. ).int()
    # result2 = torch.round( (tensor / delta_d) + n_negatives)   
    return torch.round( tensor / delta_d ).int() + n_negatives

    # result = torch.zeros_like( tensor, dtype=torch.int32 ).to( device = tensor.device )
    # for boundary in bucket_boundaries:
    #     result += (tensor > boundary).int()
    # breakpoint()
    # return result

def TorchFFTConv2d(a, K):
    """
    FFT tensor convolution of image a with kernel K 
    
    Args:
        a (torch.Tensor):   1-channel Image as tensor with at least 2 dimensions. 
                            Dimensions -2 & -1 are spatial dimensions and all other
                            dimensions are assumed to be batch dimensions
        K (torch.Tensor):   1-channel kernel as tensor with at least 2 dimensions.
    Return:
        Absolute value of the convolution of image a with kernel K 
    """
    K = torch_fft.rfft2(K)
    a = torch_fft.rfft2(a)

    img_conv = TorchComplexMul(K, a)
    img_conv = torch_fft.irfft2(img_conv)
    
    return (img_conv**2).sqrt().cpu()