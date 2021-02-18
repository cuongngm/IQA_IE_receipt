# RIVF2021 MC-OCR Solutions

# Setup folder
```
python setup.py
# copy folder dataset provided (mcocr_private_test_data, mcocr_train_data,mcocr_val_data, warmup_data) to dataset/ folder
```
# Prepare data
(detail description was writen in file):
```bash
cd prepare_data
python task1_diqa.py  # for task 1
python task2.py  # for task 2 detect
python task2_ocr.py  # for task 2 ocr
```

## Install
```bash
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## Train 
For task 1:
```bash
cd task1_diqa
python train_recognition.py --batch_size 32 --epochs 100
```
For task 2 detect using Yolov5:

download pretrain yolov5s (v4.0) in:
[here](https://github.com/ultralytics/yolov5/releases) and put them in task2 folder
```bash
cd task2
python train.py --img 1024 --batch 8 --epochs 100 --weights yolov5s.pt --data receipt.yaml

# weights and logs saved in runs/train/
```
For task 2 ocr using library VietOCR:
```bash
cd task2
python train_recognition.py
```
## Checkpoints
| Model | size(MB) |
|---------- |------ |
| [Task1_DIQA](https://drive.google.com/file/d/1EXlNxu3gpGqX00i479pV__337emazZHI/view?usp=sharing)    |5.2    
| [Task2_YOLOv5](https://drive.google.com/file/d/1G0WfRoj-frDxc6dPg-IakHUOsQh1q8A_/view?usp=sharing)    |14.8     
| [Task2_Seq2seq](https://drive.google.com/file/d/1OI7tTHvcDtXPTrv_2tF-EGeDNakAHfX7/view?usp=sharing)    |89.6     
| [Task2_Transformer](https://drive.google.com/file/d/1-oo33uZYTSi8hd1YRRIBVZY-ayt-aQ5e/view?usp=sharing)    |151.8 

Download and put them in weights folder.
## Predict
For task 1:
```bash
cd task1_diqa
python predict.py
```
For task 2:
```bash
cd task2
python main.py
```
## Submit results
```bash
python submit.py
```
