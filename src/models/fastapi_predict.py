from fastapi import UploadFile, File, FastAPI
from google.cloud import storage
from http import HTTPStatus
import torch
from torchvision import models
import io
from torch import nn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def cv_model(data: UploadFile = File(...)):
    try:
        content = await data.read()

        output = predict(content)
        response = {
            "output": output,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }
        
    except BaseException as e:
        response = {
            "output": str(e),
            "message": HTTPStatus.BAD_REQUEST.phrase,
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    return response


def predict(image):
   # get model
    client = storage.Client.from_service_account_json("vast-flight-374515-36a0dca1ba5d.json")
    bucket = client.get_bucket("trained-models-bucket")
    blob = bucket.blob("trained-model")
    model = torch.nn.Sequential(
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
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = models.resnet50(pretrained=True)
    
    #for param in model.parameters():
    #    param.requires_grad = False   
    
    #model.fc = nn.Sequential(
    #    nn.Linear(2048, 128),
    #    nn.ReLU(inplace=True),
    #    nn.Linear(128, 2)).to(device)

    with blob.open("rb") as f:
        buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer))

   # run img on model
    output = model(image)

    return output