from model import init_model
from predict_model import evaluate

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config):
    hparams = config.experiment
    print("Training day and night")
    print("learning rate: ", hparams["lr"])
    print("Training for {} epochs".format(hparams["epochs"]))
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(hparams["seed"])

    train_set = torch.load("data/processed/train_dataset")
    train_set = torch.DataLoader(train_set, batch_size = hparams["batch_size"], shuffle=True)

    model = init_model(pretrain=hparams["pretrain"])

    if hparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif hparams["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(hparams["epochs"]):
        log.info("Epoch: {}/{}".format(e+1,hparams["epochs"]))
        running_loss = 0
        for images,labels in train_set:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            log.info(f"Training loss: {running_loss/len(train_set)}")
            torch.save(model.state_dict(), 'models/trained_model.pth')

if __name__ == "__main__":
    train()
