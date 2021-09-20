import os
import torch 
from torch.utils.data import Dataset
from utils import read_image, preprocess_img

class mirflickrDataset(Dataset):

    def __init__(self,
                 set_path,
                 dims=(256,256)
                 ):
        self.set_path = set_path
        self.dims = dims

    def __len__(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dirpath, self.set_path, 'groundtruth')

        if not os.path.isdir(path):
            raise OSError(f'{path} does not exists, invalid path provided during initialization')

        num_files = len([f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))])
        return num_files

    def __getitem__(self, idx):
        gs_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.set_path, f"grayscale/gs{idx}.jpg")
        gt_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.set_path, f"groundtruth/gt{idx}.jpg")

        gs, _ = read_image(gs_loc)
        
        gt, _ = read_image(gt_loc)

        grayscale = preprocess_img(gs, dims=self.dims)
        rgb = preprocess_img(gt, dims=self.dims)

        return (grayscale[1], rgb[2])
