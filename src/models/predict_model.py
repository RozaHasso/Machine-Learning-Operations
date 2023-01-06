from model import CNN

import argparse
import torch

def evaluate(model_checkpoint, batch_size):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    criterion = torch.nn.CrossEntropyLoss()

    # TODO: Implement evaluation logic here
    model = CNN()
    test_set = torch.load("data/processed/test_dataset")
    test_set = torch.DataLoader(test_set, batch_size = batch_size)

    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    running_loss = 0
    for images,labels in test_set:
        images = images.view(images.shape[0],-1)
        output = model(images)
        loss = criterion(output,labels)
        running_loss += loss.item()
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))        
    print(f'Accuracy: {torch.mean(accuracy.item()*100)}% \nAverage loss: {torch.mean(running_loss)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Path to model checkpoint")
    parser.add_argument("--bs", type=int, help="Batch size for training loader", default=16)
    args = parser.parse_args() 
    evaluate(args.m, args.bs)