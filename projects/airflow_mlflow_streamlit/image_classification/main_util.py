from projects.airflow_mlflow_streamlit.image_classification.model_utils import initiate_model,save_model_weights,train_model
import os
import torch
import ast
from projects.airflow_mlflow_streamlit.image_classification.dataset import create_dataset
from zipfile import ZipFile

def absoluteFilePaths(directory):
    fn_list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            fn_list.append(os.path.abspath(os.path.join(dirpath, f)))

    return fn_list

def unzip_file():
    zipfile = '/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/zip_files/dataset.zip'

    if os.path.isfile(zipfile):
        with ZipFile(zipfile, 'r') as zObject: 
            zObject.extractall(path='/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards') 
        
        #delete zip file
        os.remove(zipfile)
        
    else:
        pass

def train_model():
    with open('/Users/WW/airflow/projects/airflow_mlflow_streamlit/class_idx.txt') as f:
        classes = f.read()

    classes= ast.literal_eval(classes)

    #instantiate model
    model_ = initiate_model()

    image_datasets,dataloader = create_dataset(train_path='/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/train',
                                               val_path='/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/valid',train_batch=1024,val_batch=32)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_.fc.parameters())

    model_trained = train_model(model_, image_datasets, dataloader,  criterion, optimizer, num_epochs=1)

    save_model_weights(model_trained)

