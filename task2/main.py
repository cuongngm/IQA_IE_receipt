"""
this file to predict for task 2
output: result/task2.txt
"""
import os
import cv2
import numpy as np
from predict import detect
from models.experimental import attempt_load
from recognition import RecognitionReceipt
# import sys
import torch
# sys.path.append('task2/')
config = {
        'weights': ['../weights/bestv5s.pt'],
        'imgsz': 1024,
        'conf_thres': 0.66,
        'iou_thres': 0.6,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    }


def sort_by_line(boxes):
    all_same_row = []
    same_row = [boxes[0]]
    for i in range(len(boxes) - 1):
        if boxes[i+1][1] - boxes[i][1] < 30:
            same_row.append(boxes[i+1])
        else:
            all_same_row.append(same_row)
            same_row = [boxes[i+1]]
    all_same_row.append(same_row)
    sort_same_row = []
    for row in all_same_row:
        row.sort(key=lambda x: x[0])
        for box in row:
            sort_same_row.append(box)
    return sort_same_row


def gen_box(boxes, cls, scores):
    seller = []
    address = []
    time = []
    cost = []
    if len(boxes) != 0:
        box_info = []
        for (box, cl) in zip(boxes.tolist(), cls.tolist()):
            box_info.append([box, cl])
            if cl == 0:
                seller.append(box)
            elif cl == 1:
                address.append(box)
            elif cl == 2:
                time.append(box)
            elif cl == 3:
                cost.append(box)
    seller.sort(key=lambda x: x[1])
    address.sort(key=lambda x: x[1])
    time.sort(key=lambda x: x[1])
    if len(time) != 0:
        time = sort_by_line(time)
    cost.sort(key=lambda x: x[0])
    return seller, address, time, cost


if __name__ == '__main__':
    import time
    import pandas as pd
    start = time.time()
    # load model
    recognition = RecognitionReceipt()
    # detect_model = torch.load(config['weights'], map_location=config['device'])['model'].float()
    detect_model = attempt_load(config['weights'], map_location=config['device'])
    detect_model.to(config['device']).eval()
    df = pd.read_csv('../dataset/mcocr_private_test_data/mcocr_test_samples_df.csv')
    list_id = df['img_id'].tolist()
    directory = '../dataset/mcocr_private_test_data/test_images'
    dic = {0: 'seller', 1: 'address', 2: 'timestamps', 3: 'total_cost'}
    submit = 'img_id,anno_texts' + '\n'
    for index, filename in enumerate(list_id):
        # for filename in os.listdir(directory):
        # filename = 'mcocr_private_145120sidqj.jpg'
        im0 = cv2.imread(os.path.join(directory, filename))
        im_h, im_w = im0.shape[:2]
        im0_right90 = cv2.rotate(im0, cv2.cv2.ROTATE_90_CLOCKWISE)
        im0_left90 = cv2.rotate(im0, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        im0_180 = cv2.rotate(im0, cv2.cv2.ROTATE_180)
        boxes_right90, scores_right90, cls_right90 = detect(im0_right90, config['imgsz'], detect_model,config['device'],
                                                            config['conf_thres'], config['iou_thres'])
        boxes_left90, scores_left90, cls_left90 = detect(im0_left90, config['imgsz'], detect_model, config['device'],
                                                         config['conf_thres'], config['iou_thres'])
        boxes_180, scores_180, cls_180 = detect(im0_180, config['imgsz'], detect_model, config['device'],
                                                config['conf_thres'], config['iou_thres'])
        boxes, scores, cls = detect(im0, config['imgsz'], detect_model, config['device'],
                                    config['conf_thres'], config['iou_thres'])
        compare = np.array([(len(boxes), len(boxes_right90), len(boxes_left90), len(boxes_180))])
        if np.argmax(compare) == 1:
            im0 = im0_right90
            boxes, scores, cls = boxes_right90, scores_right90, cls_right90
        elif np.argmax(compare) == 2:
            im0 = im0_left90
            boxes, scores, cls = boxes_left90, scores_left90, cls_left90
        elif np.argmax(compare) == 3:
            im0 = im0_180
            boxes, scores, cls = boxes_180, scores_180, cls_180

        seller, address, time, cost = gen_box(boxes, cls, scores)
        num = len(seller) + len(address) + len(time) + len(cost)
        """
        if len(boxes) == 0:
            im0 = cv2.rotate(im0, cv2.cv2.ROTATE_180)
            boxes, scores, cls = detect(im0, config['imgsz'], detect_model, config['device'],
                                        config['conf_thres'], config['iou_thres'])
            seller, address, time, cost = gen_box(boxes, cls, scores)
        else:
            if len(seller) != 0:
                if seller[0][1] > im0.shape[0] * 0.5:
                    im0 = cv2.rotate(im0, cv2.cv2.ROTATE_180)
                    boxes, scores, cls = detect(im0, config['imgsz'], detect_model, config['device'],
                                            config['conf_thres'], config['iou_thres'])
                    seller, address, time, cost = gen_box(boxes, cls, scores)
        """

        fields = [seller] + [address] + [time] + [cost]
        all_text = []

        for field in fields:
            box_field = []
            for box in field:
                box_im = im0[box[1]: box[3], box[0]: box[2]]
                box_field.append(box_im)
            text = recognition.recognition(box_field)
            text = ' '.join(text)
            all_text.append(text)

        all_text_final = '|||'.join(all_text)
        print(filename, all_text_final)
        submit += filename + ',' + all_text_final + '\n'
        if index == 5:
            break
        """
        # visualize
        copy = im0.copy()
        for (box, score, cl) in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
            print(box, score, dic[cl])
            copy = cv2.rectangle(copy, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            copy = cv2.putText(copy, dic[cl] + str(round(score, 2)), (box[0] + 10, box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, thickness=2, color=cv2.LINE_AA)
        copy = cv2.resize(copy, (500, 700))
        cv2.imshow('rs', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
    with open('../result/task2.txt', 'w') as file:
        file.write(submit)
