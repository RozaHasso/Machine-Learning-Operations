from model import CNN
import torch
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import argparse

def train(lr,epochs,batch_size, optimizer):
    print("Training day and night")
    print(lr)
    print(epochs)

    model = CNN()
    train_set = torch.load("data/processed/train_dataset")
    train_set = DataLoader(train_set, batch_size = batch_size)

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for e in range(epochs):
        print("Epoch: {}/{}".format(e+1,epochs))
        running_loss = 0
        for images,labels in train_set:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            losses.append(loss)
        else:
            print(f"Training loss: {running_loss}")
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", default=1e-4)
    parser.add_argument("--e", type=int, help="Number of epochs to train for", default=5)
    parser.add_argument("--bs", type=int, help="Batch size for training loader", default=16)
    parser.add_argument("--o", type=str, help="Optimizer", default="SGD")
    args = parser.parse_args() 
    train(args.lr, args.e, args.bs, args.o)
