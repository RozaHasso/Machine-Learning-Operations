
import sys
from src.data.make_dataset import DataModel
import torchvision
from torchvision import transforms
import os

dm = DataModel("./data/raw", "./data/processed")
dataloader = dm.get_dataloader()

print(dataloader.__getitem__(0)[0])