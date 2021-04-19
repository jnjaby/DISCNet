import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__+'/../..'))
from utils import img_utils, torch_utils, torch_fft


def psf_convolve(data_path, subset='', psf_type='ZTE_new', device=torch.device('cpu'), pad_size=200):
    """
    Convolution of ground truth image and PSF to simulate UDC image
    
    Args:
        data_path (str) :   Path to dataset root directory
                
        psf_type (str) :    Type of PSF from screen version 
                            Options: ['ZTE_new', 'A5', 'Sony'] | Default: 'ZTE_new'
                            
        device (torch.device) :     Device used for PyTorch tensors 
                                    Options: [torch.device('cpu'), torch.device('cuda:0'), torch.device('cuda:1'), ...]
                                    Default: torch.device('cpu)
    """
    
    print('Device:', device)
    
    ## DIRECTORIES
    psf_path = os.path.join(data_path, 'PSF/{}'.format(psf_type))
    in_path = os.path.join(data_path, 'synthetic_data/GT/{}/'.format(subset))

    for pos in range(1, 10):
        out_path = os.path.join(data_path, 'synthetic_data/input/{}_{}/{}/'.format(psf_type, pos, subset))
        os.makedirs(out_path, exist_ok=True)
        print('IN_PATH:', in_path)
        print('OUT_PATH:', out_path)

        # Load PSF
        psf = np.load(os.path.join(psf_path,'{}_psf_{}.npy'.format(psf_type, pos))).astype('float32')
        print(os.path.join(psf_path,'{}_psf_{}.npy'.format(psf_type, pos)))
        assert psf is not None, ('No PSF file found.')

        
        filenames = [f for f in sorted(os.listdir(in_path)) if f.endswith('.npy')]
        if len(filenames) == 0:
            raise Exception('No .npy files found in "{}" A subset argument may be required.'.format(in_path))
        
        # Go through files in folder
        for file in tqdm(filenames):
            # Load ground truth images
            img = np.load(os.path.join(in_path, file))
            
            # Pad or crop PSF if shape not the same as input image
            h, w, _ = img.shape
            pad_img = img_utils.pad_edges(img, (h + pad_size*2, w + pad_size*2))

            psf_matched = psf
            if psf_matched.shape[0] != pad_img.shape[0] or psf_matched.shape[1] != pad_img.shape[1]:
                psf_matched = img_utils.match_dim(psf_matched, pad_img.shape[:2])
            
            # FFT Convolution of image and PSF
            img_sim = np.zeros_like(img)
            for c in range(3):
                img_sim[..., c] = img_utils.center_crop(torch_utils.TorchFFTConv2d(torch.tensor(pad_img[..., c]).to(device),
                                        torch.tensor(psf_matched[..., c]).to(device)).numpy(), (h, w))
                img_sim = np.clip(img_sim, a_min=0, a_max=500)
            # Save output numpy file
            np.save(os.path.join(out_path, file), img_sim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Specify path to dataset root directory eg. "../"', 
                        type=str, default='.')
        
    parser.add_argument('--psf_type', help='PSF from screen version | Options: ["ZTE_new"]', 
                        type=str, default='ZTE_new')
    
    parser.add_argument('--device', help='Device used for PyTorch tensors | Options: ["cpu", "cuda:0", "cuda:1", ...]', 
                        type=str, default='cpu')

    parser.add_argument('--pad_size', help='Padding size set to avoid boundary effects caused by FFT', 
                        type=int, default=200)

    args = parser.parse_args()
    
    device = torch.device(args.device if (torch.cuda.is_available() and args.device[:3] == 'cuda') else "cpu")
    
    # For training subset.
    psf_convolve(data_path=args.data_path, subset="train", 
                 psf_type=args.psf_type, device=device, pad_size=args.pad_size)

    # For test subset.
    psf_convolve(data_path=args.data_path, subset="test", 
                 psf_type=args.psf_type, device=device, pad_size=args.pad_size)