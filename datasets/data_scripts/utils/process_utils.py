import numpy as np
import cv2

from os import path as osp
from tqdm import tqdm
import shutil
import sys


def imwrite(path, img):
    img = (img[:, :, [2,1,0]] * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, img)


def imread(path):
    img = cv2.imread(path)
    return img[:, :, [2, 1, 0]]


def tmap(x):
    '''
    Tone mapping algorithm. Refered to as simple tone-mapped domain.
    '''
    return x / (x + 0.25)


def ccm_info_get(ccm_txt):
    with open(ccm_txt) as fi:
        for line in fi:
            (key, val) = line.split(':')
            val_list = val.split()
            ccm = [np.float32(v) for v in val_list]
            ccm = np.array(ccm)
            ccm = ccm.reshape((3, 3))
    return ccm


def ccmProcess_rgb(img, ccm):
    '''
    Input images are in RGB domain.
    '''

    new_img = img.copy()
    new_img[:,:,0] = ccm[0,0] * img[:,:,0] + ccm[0,1]* img[:,:,1] + \
                        ccm[0,2] * img[:,:,2]
    new_img[:,:,1] = ccm[1,0] * img[:,:,0] + ccm[1,1]* img[:,:,1] + \
                        ccm[1,2] * img[:,:,2]
    new_img[:,:,2] = ccm[2,0] * img[:,:,0] + ccm[2,1]* img[:,:,1] + \
                        ccm[2,2] * img[:,:,2]
    return new_img


def cc_img(img, ccm):
    '''
    Color correct an image given corresponding matrix.
    Assume images in linear domain.
    '''
    
    # clip to fit ZTE sensor
    img = np.clip(img, 0, 16.0)
    img_cc = ccmProcess_rgb(img, ccm)
    img_cc = np.clip(img_cc, 0, 16)
    return img_cc


def WB(img, ref):
    '''
    Simple white balance algorithm to copy color from reference image.
    Assume both images range [0, 1].
    '''

    balanced_img = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        balanced_img[:, :, c] = img[:, :, c] / img[:, :, c].sum() * ref[:, :, c].sum()
    balanced_img = np.clip(balanced_img, 0, 1)
    return balanced_img


def simple_to_linear(img, linear_max=500):
    '''
    From simple tone-mapped domain to linear domain.
    '''
    img = np.clip(img, 0, tmap(linear_max))
    img = img / (4 * (1-img))
    return img


def linear_to_gamma(img, linear_max=12):
    A = 1 / linear_max**(1/2.8)

    img = np.clip(img, 0, linear_max)
    img = A*(img**(1/2.8))
    return img


def contrast(img, limit=1.0):
    '''
    Apply contrast enhancement. Tune argument "limit" to adjust.
    '''
    img = (img[:, :, [2,1,0]] * 255.0).round().astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2[:, :, [2,1,0]] / 255.0
