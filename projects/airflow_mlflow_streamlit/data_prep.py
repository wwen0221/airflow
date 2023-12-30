import json
import os
import pandas as pd    
import shutil
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import shutil


# jsonObj = pd.read_json(path_or_buf='/Users/WW/Downloads/input_output.jsonl', lines=True)

# doc = jsonObj['output-answer'].tolist()

# full_doc = ''

# for sentence in doc:
#     full_doc += sentence+' '

# with open('/Users/WW/Supahands/llm/llama-documentation.txt','w') as f:
#     f.write(full_doc)


# csv_files = os.listdir('/Users/WW/Supahands/llm/datasets/csv')
# for csv in csv_files:
#     fn = csv.split('.csv')[0]
#     with open(f'/Users/WW/Supahands/llm/datasets/txt/{fn}','w') as f:
#         df = pd.read_csv(f'/Users/WW/Supahands/llm/datasets/csv/{csv}',usecols=['Answers'])
#         ans = df['Answers'].tolist()
#         for sent in ans:
#             f.write(sent)

# folders = ['train','val','test']

# main_dir = '/Users/WW/Side_projects/mlops/mlops/dataset/handwritting'
# for folder in folders:
#     current_files = os.listdir(f'{main_dir}/{folder}')
#     for file in current_files:
#         cls_name = file.split('@')[0]
#         img_name = file.split('@')[-1]
#         Path(f'{main_dir}/{folder}/{cls_name}').mkdir(parents=True, exist_ok=True)
#         shutil.move(f'{main_dir}/{folder}/{file}',f'{main_dir}/{folder}/{cls_name}/{img_name}')

if __name__ == "__main__":
    # df = pd.read_csv('dataset/cards/cards.csv')
    # df = df.loc[df['data set']=='train']

    # classes = df['labels'].unique().tolist()
    # print(classes)

    # with open('class_idx.txt','w') as f:
    #     f.write(str(classes))

    dir = '/Users/WW/Side_projects/mlops/mlops/dataset/cards/train'
    wanted_list = ['001.jpg','002.jpg','003.jpg','004.jpg','005.jpg','006.jpg','007.jpg','008.jpg',
                   '009.jpg','010.jpg','011.jpg','012.jpg','013.jpg','014.jpg','015.jpg','016.jpg',
                   '017.jpg','018.jpg','019.jpg','020.jpg','021.jpg','022.jpg','023.jpg','024.jpg',
                   '025.jpg','026.jpg','027.jpg','028.jpg','029.jpg','030.jpg','031.jpg','032.jpg',
                   '033.jpg','034.jpg','035.jpg','036.jpg','037.jpg','038.jpg','039.jpg','040.jpg',
                   '041.jpg','042.jpg','043.jpg','044.jpg','045.jpg','046.jpg','047.jpg','048.jpg',
                   '049.jpg','050.jpg']
    
    all_folders = os.listdir(dir)

    for folder in all_folders:
        print(folder)
        try:
            files_in_folder = os.listdir(dir+'/'+folder)
            remove_list = list(set(files_in_folder)-set(wanted_list))
            if len(remove_list) > 0:
                for fn in remove_list:
                    os.remove(f'{dir}/{folder}/{fn}')
        except:
            continue