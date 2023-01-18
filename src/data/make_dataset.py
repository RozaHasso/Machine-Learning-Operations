# -*- coding: utf-8 -*-
import logging
import os
import shutil
import numpy as np
import torch
import torchvision
from torchvision import transforms
from skimage.transform import resize
import imageio
from skimage.color import rgb2gray
import click
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, train:bool, input_filepath: str, output_filepath) -> None:
        self.train = train
        self.work_dir = os.getcwd()
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        
        if self.output_filepath:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass
        split = "train" if self.train else "test"

        cats_dir_path = f"{input_filepath}/{split}/cats"
        dogs_dir_path = f"{input_filepath}/{split}/dogs"
        cats_filelist = os.listdir(cats_dir_path)
        dogs_filelist = os.listdir(dogs_dir_path)
        data=[]
        targets=[]
        image_size=(224,224)
        for fname in cats_filelist:
            gray_img= rgb2gray(imageio.imread(f"{cats_dir_path}/{fname}"))
            img = resize(gray_img, output_shape=image_size, mode='reflect', anti_aliasing=True)
            data.append(img) 
            targets.append(self.get_label_from_fname(fname))
        for fname in dogs_filelist:
            img = imageio.imread(f"{dogs_dir_path}/{fname}")
            if img.ndim == 3:
                img= rgb2gray(imageio.imread(f"{dogs_dir_path}/{fname}"))
            img = resize(img, output_shape=image_size, mode='reflect', anti_aliasing=True)
            data.append(img) 
            targets.append(self.get_label_from_fname(fname))

        self.data=torch.Tensor(data).reshape(-1, 1, image_size[0], image_size[1])
        self.targets=torch.LongTensor(targets)
        print(self.data.shape)
        if self.output_filepath:
            self.save_preprocessed()

    def get_label_from_fname(self, fname):
        if fname.split("_")[0] == 'cat':
            return 0
        else : 
            return 1
        

    def load_preprocessed(self):
        split = "train" if self.train else "test"
        try:
            self.data, self.targets = torch.load(
                f"{self.output_filepath}/{split}_processed.pt")
            print(self.data.shape)
            print(self.targets.shape)
        except:
            raise ValueError("No preprocessed files found")
    
    def save_preprocessed(self): 
        split = "train" if self.train else "test"
        torch.save([self.data, self.targets],
                   f"{self.output_filepath}/{split}_processed.pt")
    
    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int):
        return self.data[idx].float(), self.targets[idx]

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

