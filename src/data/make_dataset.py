# -*- coding: utf-8 -*-
import logging
import os
import shutil
import torchvision
from torchvision import transforms

import click


class DataModel():
    def __init__(self, input_filepath: str, output_filepath) -> None:
        self.work_dir = os.getcwd()
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.create_dirs()
        self.restructure_data()

    def create_dirs(self):
        try:
            os.mkdir(self.output_filepath)
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/train")
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/test")
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/train/cat")
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/train/dog")
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/test/dog")
        except:
            None
        try:
            os.mkdir(f"{self.output_filepath}/test/cat")
        except:
            None

    def restructure_data(self):
        self.create_dirs()
        dst_parent_dir = self.input_filepath+"/processed"
        data_dir_list = os.listdir(self.input_filepath)
        for data_dir in data_dir_list:
            if data_dir == 'train' or data_dir == 'test':
                data_sub_dir_path = self.input_filepath + "/" + data_dir
                data_file_list = os.listdir(data_sub_dir_path)
                for d_name in data_file_list:
                    if d_name == 'cats' or d_name == 'dogs':
                        for f_name in os.listdir(data_sub_dir_path + '/' + d_name):
                            category = 'cat' if d_name == 'cats' else 'dog'
                            dst_type_dir = f"{self.output_filepath}/{data_dir}/{category}/"
                            shutil.copyfile(data_sub_dir_path+'/' +
                                            d_name+'/'+f_name, dst_type_dir+f_name)

    def get_dataloader(self):
        return torchvision.datasets.ImageFolder("./data/processed", transform =  transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    # restructure_data(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
