import sys, os
sys.path.insert(1, '../models/')
import torch
from torch.utils.data import DataLoader
from PIL import Image

'''Testing the shape of each datapoint'''
def test_shape():
    train_set = torch.load("data/processed/train_dataset")
    train_set = DataLoader(train_set)

    test_set = torch.load("data/processed/test_dataset")
    test_set = DataLoader(test_set)

    _check_shape(train_set)
    _check_shape(test_set)

def _check_shape(data):
    for images, labels in data:
        assert images.shape == (1, 3, 224, 224)
        assert labels.shape == (1,)

'''Testing the format of each raw image'''
def test_format():
    for image_file in os.listdir("data/raw/train/cats"):
        with Image.open(os.path.join("data/raw/train/cats", image_file)) as img:
            _check_format(img)

    for image_file in os.listdir("data/raw/train/dogs"):
        with Image.open(os.path.join("data/raw/train/dogs", image_file)) as img:
            _check_format(img)

    for image_file in os.listdir("data/raw/test/cats"):
        with Image.open(os.path.join("data/raw/test/cats", image_file)) as img:
            _check_format(img)

    for image_file in os.listdir("data/raw/test/dogs"):
        with Image.open(os.path.join("data/raw/test/dogs", image_file)) as img:
            _check_format(img)

def _check_format(image):
    assert image.format == "JPEG", f"{image} is not in jpeg format"