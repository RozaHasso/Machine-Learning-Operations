import sys

import torch
sys.path.insert(1, '../models/')

train_set = torch.load("data/processed/train_dataset")
test_set = torch.load("data/processed/test_dataset")

'''Testing the length of the dataset'''
def test_length():

    assert len(train_set) == 557
    assert len(test_set) == 140

'''Testing the shape of each datapoint'''
def test_shape():

    _check_shape(train_set)
    _check_shape(test_set)

def _check_shape(data):
    for images, labels in data:
        assert images.shape == (3, 224, 224)
        assert isinstance(labels, int)
        assert labels in range(2)