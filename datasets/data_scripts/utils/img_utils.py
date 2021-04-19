#### Standard Library Imports

#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def match_dim(data, dim):
    """
    Resize image dimensions using crop or padding instead of up/down-sampling.

    Args:
        data (np.array): single channel or 3 channel image array. 
            If 3-channel array (i.e. RGB), dimension -1 should be channels.
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    # Pad outer regions of detector
    if data.shape[0] < dim[0] or data.shape[1] < dim[1]:
        data = pad_edges(data, dim[:2])
    # Crop out edge regions outside detector dimensions     
    if data.shape[0] > dim[0] or data.shape[1] > dim[1]:
        data = center_crop(data, dim[:2])
    return data

def pad_edges(data, dim):
    """
    Pads H, W dimensions on outer edges to match desired dimensions if input dimension is lesser.
    If difference between input and output dimension shape is odd, 
    an additional pixel is added on the bottom/right padding.

    Args:
        data (np.array): single channel or 3 channel (i.e. RGB) image array. 
            If 3-channel array, dimension -1 should be channels
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    pad_h, pad_w = [max(dim[0] - data.shape[0], 0), 
                    max(dim[1] - data.shape[1], 0)]
    pad_top = pad_bot = pad_h // 2
    pad_left = pad_right = pad_w // 2
    
    if pad_h % 2 != 0:
        pad_bot += 1
    if pad_w % 2 != 0:
        pad_right += 1
    pad_tuple = ((pad_top, pad_bot), (pad_left, pad_right))
    if len(data.shape) == 3:
        pad_tuple = pad_tuple + ((0, 0),)
    return np.pad(data, pad_width=pad_tuple)

def center_crop(data, dim):
    """
    Crops center H, W dimensions to match desired dimensions if input dimension is greater.

    Args:
        data (np.array): single channel or 3-channel image array. 
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    h_start, w_start = [max(data.shape[0] - dim[0], 0) // 2,
                        max(data.shape[1] - dim[1], 0) // 2]
    h_end, w_end = [h_start + min(dim[0], data.shape[0]),
                    w_start + min(dim[1], data.shape[1])]
    return data[h_start:h_end, w_start:w_end]

