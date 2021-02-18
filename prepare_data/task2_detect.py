"""
this files to convert annotations file was provided to format to train detect field (YOLOv5)
input: file annotations (mcocr_train_df.csv)

after run, the format of data to train is:
dataset/task2/
    images/
        train/
            mcocr_public_...jpg
            mcocr_public_...jpg
        val/
            mcocr_warmup_...jpg
            mcocr_warmup_...jpg
    labels/
        train/
            mcocr_public_...txt : (format: classes, center_x, center_y, w, h)
            mcocr_public_...txt
        val/
            mcocr_warmup_...txt
            mcocr_warmup_...txt
"""
import os
import pandas as pd
import shutil as sh
import ast
import cv2


def build_train_data():
    trans = {15: 0, 16: 1, 17: 2, 18: 3}
    directory = '../dataset/mcocr_train_data/train_images/'
    df = pd.read_csv('../dataset/mcocr_train_data/mcocr_train_df.csv')
    list_img_name = list(df['img_id'])
    list_img_box = []
    for img_box in df['anno_polygons']:
        list_img_box.append(img_box[2:-2])
    list_all_box_gen = []
    for index, img_box in enumerate(list_img_box):
        list_box_gen = []
        info_box = img_box.split('}, {')
        for box in info_box:
            res = ast.literal_eval('{' + box + '}')
            try:
                bbox = res['bbox']
                category_id = res['category_id']
                list_box_gen.append([trans[int(category_id)], int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]),
                                     int(bbox[1] + bbox[3])])
            except KeyError:
                print('file name with index {} is error annotations. Img name is {}'.format(index, list_img_name[index]))
                continue
        list_all_box_gen.append(list_box_gen)

    for img_name, all_box_gen in zip(list_img_name, list_all_box_gen):
        if len(all_box_gen) == 0:
            continue
        label_gen = ''
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        for box_gen in all_box_gen:
            center_x = ((box_gen[1] + box_gen[3]) / 2) / w
            center_y = ((box_gen[2] + box_gen[4]) / 2) / h
            w_label = (box_gen[3] - box_gen[1]) / w
            h_label = (box_gen[4] - box_gen[2]) / h
            classes = box_gen[0]
            label_gen += str(classes) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' +\
                         str(w_label) + ' ' + str(h_label) + '\n'
        with open('../dataset/task2/labels/train/' + img_name.split('.')[0] + '.txt', 'w') as file:
            file.write(label_gen)
        sh.copy(directory + img_name, '../dataset/task2_detect/images/train/')


def build_val_data():
    trans = {15: 0, 16: 1, 17: 2, 18: 3}
    directory = '../dataset/warmup_data/warmup_data/'
    df = pd.read_csv('../dataset/warmup_data/warmup_train.csv')
    list_img_name = []
    for img_name in df['img_id']:
        list_img_name.append(img_name)
    list_img_box = []
    for img_box in df['anno_polygons']:
        list_img_box.append(img_box[2:-2])
    list_all_box_gen = []
    for index, img_box in enumerate(list_img_box):
        list_box_gen = []
        info_box = img_box.split('}, {')
        for box in info_box:
            res = ast.literal_eval('{' + box + '}')
            try:
                bbox = res['bbox']
                category_id = res['category_id']
                list_box_gen.append([trans[int(category_id)], int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]),
                                     int(bbox[1] + bbox[3])])
            except KeyError:
                print(
                    'file name with index {} is error annotations. Img name is {}'.format(index, list_img_name[index]))
                continue
        list_all_box_gen.append(list_box_gen)

    for img_name, all_box_gen in zip(list_img_name, list_all_box_gen):
        if len(all_box_gen) == 0:
            continue
        label_gen = ''
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        for box_gen in all_box_gen:
            center_x = ((box_gen[1] + box_gen[3]) / 2) / w
            center_y = ((box_gen[2] + box_gen[4]) / 2) / h
            w_label = (box_gen[3] - box_gen[1]) / w
            h_label = (box_gen[4] - box_gen[2]) / h
            classes = box_gen[0]
            label_gen += str(classes) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + \
                         str(w_label) + ' ' + str(h_label) + '\n'
        with open('../dataset/task2/labels/val/' + img_name.split('.')[0] + '.txt', 'w') as file:
            file.write(label_gen)
        sh.copy(directory + img_name, '../dataset/task2_detect/images/val/')


if __name__ == '__main__':
    # run function build_train_data() to build dataset train yolo
    build_train_data()
    # run function build_val_data() to build dataset validation yolo
    build_val_data()
