import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil



if __name__ == '__main__':
    df = pd.read_csv('/Users/WW/Side_projects/mlops/mlops/dataset/handwritting/label.csv')
    train, validate, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.8 * len(df)), int(.9 * len(df))])


    train_img_list = train['img_name'].tolist()
    test_img_list = test['img_name'].tolist()
    val_img_list = validate['img_name'].tolist()

    for img_fn in train_img_list:
        shutil.copy(f'dataset/handwritting/handwritten-data/synthetic-data/{img_fn}', f'dataset/handwritting/train/{img_fn}')

    for img_fn in test_img_list:
        shutil.copy(f'dataset/handwritting/handwritten-data/synthetic-data/{img_fn}', f'dataset/handwritting/test/{img_fn}')

    for img_fn in val_img_list:
        shutil.copy(f'dataset/handwritting/handwritten-data/synthetic-data/{img_fn}', f'dataset/handwritting/val/{img_fn}')