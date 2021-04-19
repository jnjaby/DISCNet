import cv2
import math
import numpy as np
import lpips
import torch

from basicsr.metrics.metric_util import reorder_image
from basicsr.data.transforms import totensor

def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC'):
    """Calculate LPIPS metric.

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: LPIPS result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)


    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1, img2 = totensor([img1, img2], bgr2rgb=False, float32=True)

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = (img1 / 255. - 0.5) * 2
    img2 = (img2 / 255. - 0.5) * 2

    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores

    metric = loss_fn_alex(img1, img2).squeeze(0).float().detach().cpu().numpy()
    return metric.mean()