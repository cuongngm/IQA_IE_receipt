import cv2
from PIL import Image
import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class RecognitionReceipt(object):
    def __init__(self):
        # self.config = Cfg.load_config_from_name("vgg_transformer")
        # self.config["weights"] = 'recognition/weights/new_transformer.pth'
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        self.config['weights'] = '../weights/weights_seq2seq.pth'
        self.config["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config["cnn"]["pretrained"] = False
        self.config["predictor"]["beamsearch"] = False
        self.detector = Predictor(self.config)

    def recognition(self, list_img):
        all_text = []
        for img in list_img:
            try:
                h, w = img.shape[:2]
                if w/h < 0.5:
                    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                s = self.detector.predict(img_pil)
                if len(s) < 3:
                    img_pil = img_pil.rotate(180)
                    s = self.detector.predict(img_pil)
                all_text.append(s)
            except:
                continue
        return all_text

    def recognition_new(self, list_img):
        all_text = []
        for img in list_img:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                s = self.detector.predict(img_pil)
                all_text.append(s)
            except:
                continue
        return all_text
