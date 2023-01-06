from model import CNN
import torch

import matplotlib.pyplot as plt
import argparse

import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config):
    print("Training day and night")
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    model = CNN()
    train_set = torch.load("data/processed/train_dataset")
    train_set = torch.DataLoader(train_set, batch_size = hparams["batch_size"])

    if hparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif hparams["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for e in range(hparams["epochs"]):
        log.info("Epoch: {}/{}".format(e+1,hparams["epochs"]))
        running_loss = 0
        for images,labels in train_set:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        else:
            log.info(f"Training loss: {running_loss/len(train_set)}")
            losses.append(running_loss)
    plt.plot(list(range(hparams["epochs"])),losses)
    plt.savefig("/reports/figures/training_curve.png")
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    train()