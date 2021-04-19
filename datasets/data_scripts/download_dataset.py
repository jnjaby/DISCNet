import argparse
import zipfile
from os import path as osp

from basicsr.utils.download import download_file_from_google_drive


def download_dataset(data_path, file_ids):

    for file_name, file_id in file_ids.items():
        save_path = osp.abspath(osp.join(data_path, file_name))
        if osp.exists(save_path):
            user_response = input(
                f'{file_name} already exist. Do you want to cover it? Y/N   ')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accpets Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            download_file_from_google_drive(file_id, save_path)

        print(f'Extracting {file_name} to {save_path}')
        unzip_dataset(file_name, data_path)

def unzip_dataset(zip_file, extract_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help='Specify path to dataset root directory', 
                        type=str, default='.')

    parser.add_argument('--dataset', type=str, default='UDC')


    args = parser.parse_args()

    file_ids = {
        'UDC': {
            'PSF.zip':  # file name
            '1tEXN9COi-tHwoLp0p73ClLtlMvQHUvJO',  # file id
            'synthetic_data.zip':
            '1ctYHU70TZlQzVwAR4yOA1G-c1gsaWqzX',
            'real_data.zip':
            '1AGwylc34JLHqaxeEMN5ODhi1daG806On'
        },
    }

    download_dataset(args.data_path, file_ids[args.dataset])
