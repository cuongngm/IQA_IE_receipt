import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import patchSifting


def default_loader(path):
    return Image.open(path).convert('L')


class DataInfoLoader:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        self.config = config
        self.img_num = len(pd.read_csv(config[dataset_name]['gt_file_path'])['img_name'])

    def get_quality_score(self):
        return pd.read_csv(self.config[self.dataset_name]['gt_file_path'])['img_quality']

    def get_img_path(self):
        return self.config[self.dataset_name]['root'] +\
               pd.read_csv(self.config[self.dataset_name]['gt_file_path'])['img_name']


"""
class DIQAData(Dataset):
    def __init__(self, dataset_name, config, data_index, status='train', load_img=default_loader):
        self.load_img = load_img
        info_loader = DataInfoLoader(dataset_name, config)
        self.images_path = info_loader.get_img_path()
        self.images_quality = info_loader.get_quality_score()
        self.img_num = info_loader.img_num
        self.status = status
        if self.status == 'train':
            self.list_index = data_index
            print('Train images: {}'.format(len(self.list_index)))

        if self.status == 'val':
            self.list_index = data_index
            print('Val images: {}'.format(len(self.list_index)))

        if self.status == 'test':
            self.list_index = data_index
            print('Test images: {}'.format(len(self.list_index)))

    def __len__(self):
        return len(self.list_index)

    def __getitem__(self, idx):
        patches = ()
        img = self.load_img(self.images_path[idx])
        patches = patches + (torch.stack(patchSifting(img)), )
        label = self.images_quality[idx]
        return patches, label
"""


class DIQADataset(Dataset):
    def __init__(self, dataset_name, config, data_index, status='train', load_img=default_loader):
        """
        :param dataset_name:
        :param config:
        :param data_index: array index of image
        :param status: train, val, test
        :param loader:
        """
        self.load_img = load_img
        info_loader = DataInfoLoader(dataset_name, config)
        self.images_path = info_loader.get_img_path()
        self.images_quality = info_loader.get_quality_score()
        self.img_num = info_loader.img_num
        self.status = status
        if self.status == 'train':
            self.list_index = data_index
            print('Train images: {}'.format(len(self.list_index)))

        if self.status == 'val':
            self.list_index = data_index
            print('Val images: {}'.format(len(self.list_index)))

        if self.status == 'test':
            self.list_index = data_index
            print('Test images: {}'.format(len(self.list_index)))

        self.patches = ()
        self.label = []
        for idx in self.list_index:
            img = self.load_img(self.images_path[idx])
            patches = patchSifting(img)
            if status == 'train':
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(self.images_quality[idx])
            else:
                self.patches = self.patches + (torch.stack(patches), )
                self.label.append(self.images_quality[idx])

    def __len__(self):
        return len(self.list_index)

    def __getitem__(self, idx):
        return self.patches[idx], torch.Tensor([self.label[idx]])


if __name__ == '__main__':
    import numpy as np
    import yaml
    import math
    import time
    from utils import DocScanner
    scan = DocScanner()
    start = time.time()
    with open('config.yaml') as f:
        config = yaml.load(f)
    dataset_name = 'bill'
    dil = DataInfoLoader(dataset_name, config)
    data_info = DataInfoLoader(dataset_name=dataset_name, config=config)
    img_num = data_info.img_num
    img_path = data_info.get_img_path()
    index = np.arange(img_num)
    np.random.shuffle(index)
    train_idx = index[0: math.floor(img_num * 0.01)]
    dataset = DIQADataset(dataset_name=dataset_name, config=config, data_index=train_idx, status='train')
    print(dataset[0])
    print(len(dataset))
