import torch
import numpy as np
import cv2
from task2.models.experimental import attempt_load
from task2.utils.datasets import letterbox
from task2.utils.general import non_max_suppression, scale_coords


def detect(im0, imgsz, model, device, conf_thres, iou_thres):
    img = letterbox(im0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # 3 x H x W
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    # apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    boxes = []
    scores = []
    cls_arr = []
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
            scores.append(conf)
            cls_arr.append(cls.numpy().astype(np.int))
    return np.array(boxes), np.array(scores), np.array(cls_arr)


if __name__ == '__main__':
    import os
    weights = ['../weights/bestv5s.pt']
    dir = '../dataset/mcocr_private_test_data/test_images/'
    imgsz = 1024
    conf_thres = 0.66
    iou_thres = 0.6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(weights, map_location=device)['model'].float()
    model = attempt_load(weights, map_location=device)
    model.to(device).eval()
    dic = {0: 'seller', 1: 'address', 2: 'timestamps', 3: 'total'}
    for filename in os.listdir(dir):
        # im0 = cv2.imread('result.jpg')
        im0 = cv2.imread(dir + filename)
        im_h, im_w = im0.shape[:2]
        if im_h < im_w:
            im0 = cv2.rotate(im0, cv2.cv2.ROTATE_90_CLOCKWISE)
        boxes, scores, cls = detect(im0, imgsz, model, device, conf_thres, iou_thres)
        for (box, cl, score) in zip(boxes.tolist(), cls.tolist(), scores.tolist()):
            im0 = cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            im0 = cv2.putText(im0, dic[cl] + str(round(score, 2)), (box[0]+10, box[1]+10), cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1, thickness=2, color=cv2.LINE_AA)
        cv2.imshow('rs', im0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
