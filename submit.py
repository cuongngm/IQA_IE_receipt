"""
import pandas as pd
import os
from zipfile import ZipFile
df1 = pd.read_csv('result/task1_image_quality.txt')
df2 = pd.read_csv('result/task2.txt')
df1['anno_texts'] = df2['anno_texts']
df1.to_csv('result/results.csv', index=False)
os.chdir('result/')
ZipFile('results.zip', mode='w').write('results.csv')
"""
import pandas as pd
import json
df = pd.read_csv('dataset/mcocr_train_data/mcocr_train_df.csv')
list_img_id = list(df['img_id'])
list_anno_texts = list(df['anno_texts'])
list_anno_labels = list(df['anno_labels'])
index = list_img_id.index('mcocr_public_145013aaprl.jpg')
for (img_id, texts, labels) in zip(list_img_id, list_anno_texts, list_anno_labels):
    try:
        texts = texts.split('|||')
        labels = labels.split('|||')
        dic = {}
        seller = []
        address = []
        timestamp = []
        total_cost = []
        for text, label in zip(texts, labels):
            text = text.replace(',', '')
            text = text.replace('+', ' ')
            if label.lower() == 'seller':
                seller.append(text)
            if label.lower() == 'address':
                address.append(text)
            if label.lower() == 'timestamp':
                timestamp.append(text)
            if label.lower() == 'total_cost':
                total_cost.append(text)
        seller = ', '.join(seller)
        address = ', '.join(address)
        timestamp = ', '.join(timestamp)
        total_cost = ', '.join(total_cost)
        dic['seller'] = seller
        dic['address'] = address
        dic['timestamp'] = timestamp
        dic['total_cost'] = total_cost
        # print(dic)
        with open('key/{}.json'.format(img_id.split('.')[0]), 'w') as outfile:
            json.dump(dic, outfile)
    except:
        print(img_id)
        # mcocr_public_145014iwhec.jpg
        # mcocr_public_145014jndnz.jpg
