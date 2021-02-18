"""
this files to convert annotations file was provided to format to train document image quality assessment (DIQA)
input: file annotations (mcocr_train_df.csv)
output: file txt (quality_info.txt) contain:
img_name,img_quality
mcocr_public_145013ddcph.jpg,0.635309
mcocr_public_145013fxcgs.jpg,0.774317
mcocr_public_145013clltn.jpg,0.664084
...

after run, the format of data to train is:
dataset/task1_diqa/
    image/
        mcocr_public_...jpg
        mcocr_public_...jpg
        ...
    label/
        anno_image_quality.txt
"""
import os
import pandas as pd
import shutil as sh


df = pd.read_csv('../dataset/mcocr_train_data/mcocr_train_df.csv')
list_img_name = []
list_img_quality = []
for img_name in df['img_id']:
    list_img_name.append(img_name)
for img_quality in df['anno_image_quality']:
    list_img_quality.append(img_quality)
quality_annotations = 'img_name,img_quality' + '\n'
for name, quality in zip(list_img_name, list_img_quality):
    quality_annotations += name + ',' + str(quality) + '\n'
with open('../dataset/task1_diqa/label/quality_info.txt', 'w') as file:
    file.write(quality_annotations)
for filename in os.listdir('../dataset/mcocr_train_data/train_images/'):
    sh.copy('../dataset/mcocr_train_data/train_images/' + filename, '../dataset/task1_diqa/image/')
