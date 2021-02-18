from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
config = Cfg.load_config_from_name('vgg_transformer')
#config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

dataset_params = {
    'name': 'hw',
    'data_root': '../dataset/task2_ocr/',
    'train_annotation': 'train_ocr.txt',
    'valid_annotation': 'val_ocr.txt'
}

params = {
         'print_every': 200,
         'valid_every': 15*200,
          'iters': 30000,
          'batch_size': 32,  # 16 if out mem
          'checkpoint': '../weights/weights_transformer.pth',
          'export': './weights/new_transformer.pth',
          'metrics': 10000,
}
config['pretrain']['cached'] = 'transformerorc.pth'
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
# config['weights'] = 'weights/vgg_seq2seq.pth'
trainer = Trainer(config, pretrained=True)
trainer.train()