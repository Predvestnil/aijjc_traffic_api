import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
from typing import Any, Optional, Tuple, Callable
from models import SignsClassifier
import os
import cv2
import albumentations as albu
import time
from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify, abort

app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)
run_with_ngrok(app)

def transform_image(image):
    tforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                     [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    return tforms(image).unsqueeze(0) / 255.0

def load_json_file(path: str) -> Any:
    """Loading a json file.

    :param path: path to the json file
    :return: json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


device = 'cpu'
class2label = load_json_file('class2label.json')
label2class = {label: sign_class for sign_class, label in class2label.items()}
state_dict_1 = torch.load('1.pth',map_location=torch.device('cpu'))['state_dict']
model = SignsClassifier('resnext50_32x4d',len(class2label))
model.load_state_dict(state_dict_1)
model.to(device)
model.eval()
 
@app.route("/test", methods=['POST'])
def test_method():         
  
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    im_b64 = request.json['image']

    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    img = Image.open(io.BytesIO(img_bytes))

    img = transform_image(img)
    with torch.no_grad():
        img = img.to(device)
        prediction = model(img)
        prediction = prediction.max(dim=-1)[1].cpu().detach().numpy()
        prediction = np.vectorize(label2class.get)(prediction)
    return str(prediction)
  
  
def run_server_api():
    app.run()
  
  
if __name__ == "__main__":     
    run_server_api()