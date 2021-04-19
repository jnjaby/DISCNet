import os
from pdb import Pdb, set_trace
import pdb
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()+'/../..'))
from utils.process_utils import (imread, imwrite, ccm_info_get, cc_img,
                                simple_to_linear, linear_to_gamma, WB, contrast)


def post_processing(data_path, ref_path, save_path, color_correct=True,
                    rgb_scaling=True, contrast_enhance=True):
    """
    Convolution of ground truth image and PSF to simulate UDC image
    
    Args:
        data_path (str) :       Path to data to be processed
        
        ref_path (str) :        Path to root directory that contains reference files

        save_path (str) :       Path to save post-processing images 

        color_correct (bool):   Whether to correct by Color Correction Matrix from camera

        rgb_scaling (bool):     Whether to copy color from camera output (JPEG)

        contrast_enhance (bool):Whether to apply contrast enhancement algorithm
    """

    jpg_dir = os.path.join(ref_path, 'jpg_ZTE/')
    CCM_dir = os.path.join(ref_path, 'CCM_txt/')

    os.makedirs(save_path, exist_ok=True)

    img_list = sorted([x.replace('.npy', '') for x in os.listdir(data_path) \
                        if x.endswith('.npy')])
    # import pdb; pdb.set_trace()
    assert len(img_list) != 0, (f'No npy files found in {data_path}.')


    for img_path in tqdm(img_list):
        # Load data
        jpg_img = imread(os.path.join(jpg_dir, img_path + '.jpg')) / 255.
        ccm = ccm_info_get(os.path.join(CCM_dir, img_path + '_ccm.txt'))

        # The output of network are assumed to be in simple tone-mapped domain.
        simple_img = np.load(os.path.join(data_path, img_path + '.npy'))

        # Convert to linear domain before post processing.
        post_img = simple_to_linear(simple_img)

        if color_correct:
            post_img = cc_img(post_img, ccm)
        
        # Gamma correction
        post_img = linear_to_gamma(post_img, linear_max=6) # gamma domain [0,1]
        
        if rgb_scaling:
            # The reference jpg images are gamma-corrected, requiring post images
            # in the same domain.
            post_img = WB(post_img, jpg_img)

        if contrast_enhance:
            post_img = contrast(post_img, limit=1.0)
        
        save_img = np.clip(post_img, a_min=0, a_max=1)

        imwrite(os.path.join(save_path, img_path + '.png'), save_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help='Specify path to dataset to be processed', 
                        type=str, default='.')
        
    parser.add_argument('--ref_path', help='Specify path to root directory that contains reference \
                                            files, mainly including camera output and ccm files.', 
                        type=str, default='datasets/real_data')

    parser.add_argument('--save_path', help='Specify path to save post-processing images', 
                        type=str, default='results/post_images')

    parser.add_argument('--color_correct', help='Whether to correct by Color Correction Matrix \
                                            from camera', 
                        type=bool, default=True)

    parser.add_argument('--rgb_scaling', help='Whether to copy color from camera output (JPEG)', 
                        type=bool, default=True)

    parser.add_argument('--contrast_enhance', help='Whether to apply contrast enhancement', 
                        type=bool, default=True)

    args = parser.parse_args()
    
    post_processing(data_path=args.data_path, ref_path=args.ref_path, \
                    save_path=args.save_path, color_correct=args.color_correct, \
                    rgb_scaling=args.rgb_scaling, contrast_enhance=args.contrast_enhance)
