"""
this files to convert annotations file was provided to format to train OCR (VietOCR)
input: file annotations (mcocr_train_df.csv)
output:


after run, the format of data to train is:
dataset/task2_ocr/
    img/
        mcocr_public_..._gen_0.jpg
        mcocr_public_..._gen_1.jpg
        ...
        (box image generate from image)
    train_ocr.txt
        (example:   task2_ocr/mcocr_public_..._gen_0.jpg    abcd
                    task2_ocr/mcocr_public_..._gen_1.jpg    efgh)
    val_ocr.txt
"""
import pandas as pd
import numpy as np
import ast
import cv2
from random import shuffle


df = pd.read_csv('../dataset/mcocr_train_data/mcocr_train_df.csv')
list_img_names = list(df['img_id'])
list_img_texts = list(df['anno_texts'])
list_img_boxes = []
for img_boxes in df['anno_polygons']:
    list_img_boxes.append(img_boxes[2:-2])
list_all_boxes_gen = []
for index, img_box in enumerate(list_img_boxes):
    list_box_gen = []
    info_box = img_box.split('}, {')
    for box in info_box:
        res = ast.literal_eval('{' + box + '}')
        try:
            bbox = res['bbox']
            category_id = res['category_id']
            list_box_gen.append([int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])
        except KeyError:
            print('file name with index {} is error annotations. Img name is {}'.format(index, list_img_names[index]))
            continue
    list_all_boxes_gen.append(list_box_gen)
info_ocr = []
for img_names, img_boxes, img_texts in zip(list_img_names, list_all_boxes_gen, list_img_texts):
    if len(img_boxes) == 0:
        continue
    img = cv2.imread('../dataset/mcocr_train_data/train_images/' + img_names)
    texts = img_texts.split('|||')
    for index, (box, text) in enumerate(zip(img_boxes, texts)):
        text_gen = 'img/{}.jpg'.format(img_names.split('.')[0] + '_gen_' + str(index)) + '\t' + text
        info_ocr.append(text_gen)
        img_crop = img[box[1]: box[3], box[0]: box[2]]
        cv2.imwrite('../dataset/task2_ocr/img/{}.jpg'.format(img_names.split('.')[0] + '_gen_' + str(index)), img_crop)
index = np.arange(len(info_ocr))
shuffle(index)
index_train = index[0: int(len(index) * 0.8)]
index_val = index[int(len(index) * 0.8): len(index)]
train_data = []
val_data = []
for i in index_train:
    train_data.append(info_ocr[i])
for j in index_val:
    val_data.append(info_ocr[j])
with open('../dataset/task2_ocr/train_ocr.txt', 'w') as file_train:
    for data in train_data:
        file_train.write(data + '\n')
with open('../dataset/task2_ocr/val_ocr.txt', 'w') as file_val:
    for data in val_data:
        file_val.write(data + '\n')
