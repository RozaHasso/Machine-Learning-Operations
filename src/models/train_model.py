from model import init_model
from predict_model import evaluate

import torch
from torch.utils.data import DataLoader
import wandb

import argparse
from google.cloud import storage

def train(lr,epochs,batch_size, optimizer, pretrain=False):
    torch.manual_seed(123)
    model = init_model(pretrain=pretrain)
    
    print("Training day and night")
    print("learning rate: ", lr)
    print("Training for {} epochs".format(epochs))
    
    train_set = torch.load("data/processed/train_dataset")
    train_set = DataLoader(train_set, batch_size = batch_size, shuffle=True)

    
    wandb.watch(model, log_freq=100)
    exp_info = wandb.Table(
            columns = ["Pretrained model:", "Optimizer:", "Batch size:", "Learning Rate:", "Epochs:"],
            data = [[pretrain, optimizer, batch_size, lr, epochs]])
    wandb.log({"Experiment info:":exp_info})
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        print("Epoch: {}/{}".format(e+1,epochs))
        running_loss = 0
        for batch_idx, (images,labels) in enumerate(train_set):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            with torch.no_grad():
                accuracy, eval_loss = evaluate(model, batch_size)
                print(f"Training loss: {running_loss/len(train_set)}, Accuracy: {accuracy}%, Val loss: {eval_loss}")
                wandb.log({"loss": loss, "epoch": e,
                            "inputs": wandb.Image(images),
                            "logits": wandb.Histogram(output),
                            "captions": labels,
                            "accuracy": accuracy,
                            "evaluation loss": eval_loss})

    client = storage.Client.from_service_account_json("vast-flight-374515-36a0dca1ba5d.json")
    bucket = client.get_bucket("trained-models-bucket")
    blob = bucket.blob("trained-model")

    with blob.open("wb", ignore_flush=True) as f:
        torch.save(model.state_dict(), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-e", type=int, help="Number of epochs to train for", default=5)
    parser.add_argument("-bs", type=int, help="Batch size for training loader", default=16)
    parser.add_argument("-o", type=str, help="Optimizer", default="Adam")
    parser.add_argument("-pt", type=str, help="Initialize pretrained model", default=True)
    args = parser.parse_args()
    wandb.init(config=args) 
    train(args.lr, args.e, args.bs, args.o, args.pt)
