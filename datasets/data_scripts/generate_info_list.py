import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__+'/../..'))


def generate_info_list(data_path, save_dir, psf_type='ZTE_new'):
    """
    Generate info list for both synthetic and real data. The lists are used
    to specify paths of input images and the corresponding PSF for training.
    
    Args:
        data_path (str) :    Path to dataset root directory
        
        real_path (str):    Path to real dataset

        save_dir (str):     Path to save kernel info list

        psf_type (str):     Type of PSF from screen version
    """
    
    ## DIRECTORIES
    syn_path = os.path.join(data_path, 'synthetic_data/input')
    real_path = os.path.join(data_path, 'real_data/input')
    code_path = os.path.join(data_path, 'PSF/kernel_code')
    os.makedirs(save_dir, exist_ok=True)

    # Generate info list for real data.
    real_save_path = os.path.join(save_dir, 'real_ZTE_list.txt')

    # Suppose real input images are generated with PSF at the center.
    curr_psf = os.path.abspath(os.path.join(code_path, \
                    '{}_code_{}.npy'.format(psf_type, '5')))
    assert os.path.isfile(curr_psf), ("PSF Code file not exists.")

    img_list = sorted(os.listdir(real_path))
    assert len(img_list) != 0, ("No image files found in '{}'.".format(real_path))

    print('Gererating info list for real data ...')
    with open(real_save_path, 'w') as f:
        for path in img_list:
            f.write(path + ' ' + curr_psf + ' \n')


    # Generate info list for synthetic data.
    syn_save_dir = os.path.join(save_dir, psf_type)
    os.makedirs(syn_save_dir, exist_ok=True)

    for subset in ['train', 'test']:
        for pos in range(1, 10):
            syn_save_path = os.path.join(syn_save_dir, 
                        '{}_code_{}_{}.txt'.format(psf_type, pos, subset))
            print('Gererating info list {} ...'.format(syn_save_path))

            # Specify path to corresponding PSF code.
            curr_psf = os.path.abspath(os.path.join(code_path, \
                            '{}_code_{}.npy'.format(psf_type, pos)))
            assert os.path.isfile(curr_psf), ("PSF Code file not exists.")

            in_path = os.path.join(syn_path, '{}_{}'.format(psf_type, pos), subset)
            img_list = [f for f in sorted(os.listdir(in_path)) if f.endswith('.npy')]
            assert len(img_list) != 0, ("No image files found in '{}'.".format(in_path))

            with open(syn_save_path, 'w') as f:
                # Go through files in folder
                for path in tqdm(img_list):
                    f.write(path + ' ' + curr_psf + ' \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Specify path to dataset root directory eg. "../"', 
                        type=str, default='.')

    parser.add_argument('--psf_type', help='Specify path to PSF eg. "../"', 
                        type=str, default='ZTE_new')

    parser.add_argument('--save_dir', help='Specify directory to save kernel info list', 
                        type=str, default='./PSF/kernel_info_list')

    args = parser.parse_args()
    
    generate_info_list(data_path=args.data_path, psf_type=args.psf_type, save_dir=args.save_dir)
