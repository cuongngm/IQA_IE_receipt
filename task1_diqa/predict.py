from PIL import Image
import torch
from model import DIQANet
from utils import patchSifting


class Solver:
    def __init__(self, model_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = DIQANet().to(self.device)
        model_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_dict)
        self.model = model
        self.model.eval()

    def quality_assessment(self, img_path):
        im = Image.open(img_path).convert('L')
        im = torch.stack(patchSifting(im))
        im = im.to(self.device)
        qs = self.model(im)
        qs = qs.data.squeeze(0).cpu().numpy()[:, 0].mean()
        return qs


if __name__ == '__main__':
    import os
    import pandas as pd
    # path = 'data/test/mcocr_public_145013aagqw.jpg'
    solver = Solver(model_path='../weights/DIQA-bill-lr=0.001.pth')
    # directory = 'data/val_images/'
    df = pd.read_csv('../dataset/mcocr_private_test_data/mcocr_test_samples_df.csv')
    list_id = df['img_id'].tolist()
    directory = '../dataset/mcocr_private_test_data/test_images/'
    # dic = {0: 'seller', 1: 'address', 2: 'timestamps', 3: 'total_cost'}
    list_quality_score = 'img_id,anno_image_quality' + '\n'
    for index, filename in enumerate(list_id):
        # for index, filename in enumerate(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        quality_score = solver.quality_assessment(filepath)
        list_quality_score += filename + ',' + str(quality_score) + '\n'
    with open('../result/task1_image_quality.txt', 'w') as file:
        file.write(list_quality_score)
    """
    import pandas as pd
    df = pd.read_csv('../result/task1_image_quality.txt')
    print(df.head())

        img_id  quality_score
    0  mcocr_private_145120ypcjr.jpg       0.731642
    1  mcocr_private_145120euvko.jpg       0.763412
    2  mcocr_private_145120aedhd.jpg       0.486086
    3  mcocr_private_145120rtovn.jpg       0.642891
    4  mcocr_private_145120ywkur.jpg       0.581188
    """