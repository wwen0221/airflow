from flask import Flask,render_template
import configparser
from werkzeug.utils import secure_filename
from flask import Flask, request
import json
import numpy as np
import os
from airflow_mlflow_streamlit.image_classification import model_utils
import cv2
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F

app = Flask(__name__)

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
}


@app.route('/', methods=["POST","GET"])
def main():

   return render_template('main_page.html')


@app.route('/upload', methods=["POST"])
def upload_file():
    if request.method =='POST':
        model_path = '/Users/WW/airflow/projects/airflow_mlflow_streamlit/models/image_classification/playing_card/2023_10_12_15_58.h5'
        loaded_model = model_utils.load_tl_model(model_path)

        print('called')
        '''
         file = request.FILES['filename']
         file.name           # Gives name
         file.content_type   # Gives Content type text/html etc
         file.size           # Gives file's size in byte
         file.read()         # Reads file
        '''
        file = request.files['file']
      
        if file:
            filename = secure_filename(file.filename)
            file_type = secure_filename(file.content_type)

            if 'zip' in file_type:
               file.save(os.path.join('/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/zip_files/',filename))
               return main()

            else:
               file.save(os.path.join('/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/test_imgs/',filename))

               #read image
               img_list = [Image.open(os.path.join('/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/test_imgs/',filename))]
               validation_batch = torch.stack([data_transforms['validation'](img).to('mps') for img in img_list])
               pred_logits_tensor = loaded_model(validation_batch)
               pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
               prob = np.max(pred_probs)
               idx = torch.argmax(pred_logits_tensor).item()
               
            return render_template('main_page.html',cls=idx,prob=prob)
        

    return render_template('main_page.html')


if __name__ == '__main__':
   app.run()