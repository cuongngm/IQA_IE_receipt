import pandas as pd
import os
from zipfile import ZipFile
df1 = pd.read_csv('result/task1_image_quality.txt')
df2 = pd.read_csv('result/task2.txt')
df1['anno_texts'] = df2['anno_texts']
df1.to_csv('result/results.csv', index=False)
os.chdir('result/')
ZipFile('results.zip', mode='w').write('results.csv')
