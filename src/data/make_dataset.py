# -*- coding: utf-8 -*-
import logging
import os
import torch
import torchvision
from torchvision import transforms

import click


def process_data(input_filepath, output_filepath):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_set = torchvision.datasets.ImageFolder("./data/raw/train", transform =  transforms.Compose(
                                                [transforms.Resize((224,224)), transforms.ToTensor(),
                                                normalize])
                                                )
    test_set = torchvision.datasets.ImageFolder("./data/raw/test", transform =  transforms.Compose(
                                                [transforms.Resize((224,224)), transforms.ToTensor(),
                                                normalize])
                                                )
    return train_set, test_set

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    train_set, test_set = process_data(input_filepath, output_filepath)
    torch.save(train_set, os.path.join(output_filepath,"train_dataset"))
    torch.save(test_set, os.path.join(output_filepath,"test_dataset"))
    

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
