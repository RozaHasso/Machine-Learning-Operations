from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
import numpy as np
import torch
from src.models.predict_model import get_prediction
from PIL import Image
from io import BytesIO
from fastapi.responses import HTMLResponse
import base64

app = FastAPI()

test_set = torch.load("data/processed/test_dataset")
tens =  test_set[130][0]


@app.get("/")
def read_root():
   return {"Prediction": get_prediction(tens)}



def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/get/")
async def get_input_prediction(file: UploadFile=File(...)):
   image = await file.read()
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

   transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
   normalize
   ])
   img_tensor = transform(Image.open(BytesIO(image)))
   html_content = f"""
       <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h4>{get_prediction(img_tensor)}</h4>
             <img src="data:image/jpg;base64,{base64.b64encode(image).decode("utf-8")}" alt="alternatetext" width="300px" height="300px"> 
        </body>
    </html>
   """
   return HTMLResponse(content=html_content, status_code=200)





