import cv2
import mmcv
import torch
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.data.util import (paired_paths_from_meta_info_file,
                               paired_paths_PSF_from_meta_info_file,
                               paired_paths_from_folder,
                               paired_paths_from_lmdb)
from basicsr.utils import FileClient


class PairedImgPSFNpyDataset(data.Dataset):
    """Paired image dataset with its corresponding PSF.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc)
    and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal and vertical flips.
            use_rot (bool): Use rotation (use transposing h and w for
                implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = []
        for folder_name, folder_opt in opt['folders'].items():
            assert folder_opt['meta_info_file'] is not None, ('Only support loading image\
                        and PSF by meta info file.')
            gt_folder, lq_folder = folder_opt['dataroot_gt'], folder_opt['dataroot_lq']

            self.paths += paired_paths_PSF_from_meta_info_file(
                            [lq_folder, gt_folder], ['lq', 'gt'],
                            folder_opt['meta_info_file'], self.filename_tmpl)


    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

    def _expand_dim(self, x):
        # expand dimemsion if images are gray.
        if x.ndim == 2:
            return x[:,:,None]
        else:
            return x
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        lq_map_type = self.opt['lq_map_type']
        gt_map_type = self.opt['gt_map_type']

        crop_scale = self.opt.get('crop_scale', None)

        # Load gt and lq images. Dimension order: HWC; channel order: RGGB;
        # HDR image range: [0, +inf], float32.
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        psf_path = self.paths[index]['psf_path']
        img_gt = self.file_client.get(gt_path)
        img_lq = self.file_client.get(lq_path)
        psf_code = self.file_client.get(psf_path)

        # tone mapping
        img_lq = self._tonemap(img_lq, type=lq_map_type)
        img_gt = self._tonemap(img_gt, type=gt_map_type)

        # expand dimension
        img_gt = self._expand_dim(img_gt)
        img_lq = self._expand_dim(img_lq)

        # Rescale for random crop
        if crop_scale != None:
            h, w, _ = img_lq.shape
            img_lq = cv2.resize(
                img_lq, (int(w * crop_scale), int(h * crop_scale)), interpolation=cv2.INTER_LINEAR)
            img_gt = cv2.resize(
                img_gt, (int(w * crop_scale), int(h * crop_scale)), interpolation=cv2.INTER_LINEAR)

        # augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = totensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        psf_code = torch.from_numpy(psf_code)[..., None, None]

        return {
            'lq': img_lq,
            'gt': img_gt,
            'psf_code': psf_code,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'psf_path': psf_path,
        }

    def __len__(self):
        return len(self.paths)
