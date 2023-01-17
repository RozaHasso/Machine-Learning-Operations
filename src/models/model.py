import torch
from torchvision import models
from torch import nn

def init_model(pretrain=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pretrain:
        model = models.resnet50(pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = False   

        model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)).to(device)
    else:
        class CNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(123)
                self.model = torch.nn.Sequential(
                    #Input = 3 x 32 x 32, Output = 32 x 32 x 32
                    torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
                    torch.nn.ReLU(),
                    #Input = 32 x 32 x 32, Output = 32 x 16 x 16
                    torch.nn.MaxPool2d(kernel_size=2),

                    #Input = 32 x 16 x 16, Output = 64 x 16 x 16
                    torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
                    torch.nn.ReLU(),
                    #Input = 64 x 16 x 16, Output = 64 x 8 x 8
                    torch.nn.MaxPool2d(kernel_size=2),

                    #Input = 64 x 8 x 8, Output = 64 x 8 x 8
                    torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
                    torch.nn.ReLU(),
                    #Input = 64 x 8 x 8, Output = 64 x 4 x 4
                    torch.nn.MaxPool2d(kernel_size=2),

                    torch.nn.Flatten(),
                    torch.nn.Linear(64*4*4, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 2),
                    torch.nn.LogSoftmax(dim=1)
                )

            def forward(self, x):
                return self.model(x)
        model = CNN()
    return model