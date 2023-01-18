from model import init_model
from src.data.make_dataset import ImgDataset

import argparse
import torch

from torch.utils.data import DataLoader

def evaluate(model, batch_size):
    print("Evaluating until hitting the ceiling")
    # test_set = torch.load("data/processed/test_dataset")
    # test_set = DataLoader(test_set, batch_size = batch_size, shuffle=True)
    test_set = ImgDataset(train=False,input_filepath= "./data/raw", output_filepath="./data/processed")
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    running_loss = 0
    with torch.no_grad():
        accuracy = 0
        for images,labels in dataloader:
            output = model(images)
            loss = criterion(output,labels)
            running_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        return accuracy/len(test_set), running_loss/len(test_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Path to model checkpoint")
    parser.add_argument("--bs", type=int, help="Batch size for training loader", default=16)
    args = parser.parse_args()
    model = init_model()
    state_dict = torch.load(args.m)
    model.load_state_dict(state_dict)
    evaluate(model, args.bs)