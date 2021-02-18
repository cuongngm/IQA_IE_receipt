import argparse
import math
import numpy as np
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import DIQANet
from loader import DataInfoLoader, DIQADataset
# pytorch ignite engine
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics.metric import Metric
from sklearn.metrics import mean_squared_error


def ensure_dir(path):
    p = Path(path)
    if not p.exists():
        p.mkdir()


def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y)


def get_data_loaders(dataset_name, config, train_batch_size):
    data_info = DataInfoLoader(dataset_name, config)
    img_num = data_info.img_num
    index = np.arange(img_num)
    np.random.shuffle(index)

    # train, val, test
    train_index = index[0:math.floor(img_num*0.7)]
    val_index = index[math.floor(img_num*0.7): math.floor(img_num*0.9)]
    test_index = index[math.floor(img_num*0.9):]

    train_dataset = DIQADataset(dataset_name, config, train_index, status='train')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=4)

    val_dataset = DIQADataset(dataset_name, config, val_index, status='val')
    val_loader = DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = DIQADataset(dataset_name, config, test_index, status='test')
        test_loader = DataLoader(test_dataset)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class DIQA_Performance(Metric):
    def reset(self):
        self.label_pred = []
        self.label = []

    def update(self, output):
        y_pred, y = output
        self.label_pred.append(torch.mean(y_pred))
        self.label.append(y)

    def compute(self):
        y_pred = np.reshape(np.asarray(self.label_pred), (-1,))
        y = np.reshape(np.asarray(self.label), (-1,))
        # y_pred = np.array(self.label_pred)
        # y = np.array(self.label)
        # rmse = np.sqrt(((y_pred - y) ** 2).mean(axis=None))
        rmse = mean_squared_error(y, y_pred, squared=False)
        return rmse


class Solver:
    def __init__(self):
        self.model = DIQANet()

    def run(self, dataset_name, train_batch_size, epochs, lr, weight_decay, config, trained_model_file):
        if config['test_ratio']:
            train_loader, val_loader, test_loader = get_data_loaders(dataset_name, config, train_batch_size)
        else:
            train_loader, val_loader = get_data_loaders(dataset_name, config, train_batch_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        global best_criterion
        best_criterion = -1
        trainer = create_supervised_trainer(self.model, optimizer, loss_fn, device=device)
        evaluator = create_supervised_evaluator(self.model, metrics={'performance': DIQA_Performance()}, device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_result(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            rmse = metrics['performance']
            print('Validation result: -Epoch: {} RMSE: {:.4f}'.format(engine.state.epoch, rmse))
            global best_criterion
            global best_epoch
            # if rmse > best_criterion:
            #     best_criterion = rmse
            best_epoch = engine.state.epoch
            print('epoch:', best_epoch)
            torch.save(self.model.state_dict(), trained_model_file)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_testing_result(engine):
            if config['test_ratio'] > 0 and config['test_during_training']:
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                rmse = metrics['performance']
                print('Test result: -Epoch: {} RMSE: {:.4f}'.format(engine.state.epoch, rmse))

        @trainer.on(Events.EPOCH_COMPLETED)
        def final_testing_result(engine):
            if config['test_ratio'] > 0:
                self.model.load_state_dict(torch.load(trained_model_file))
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                rmse = metrics['performance']
                global best_epoch
                print('Final test result - Epoch: {} RMSE: {:.4f}'.format(best_epoch, rmse))
        trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(description='Pytorch DIQA model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size training')
    parser.add_argument('--epochs', type=int, default=100, help='epoch training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('--ds_name', type=str, default='bill', help='name of dataset')
    parser.add_argument('--pretrained', type=str, default='../weights/', help='load pretrained model')
    parser.add_argument('--saved', type=str, default='saved/', help='path to save model')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    ensure_dir(args.pretrained)
    ensure_dir(args.saved)
    trained_model_file = os.path.join(args.pretrained, 'DIQA-{}-lr={}.pth'.format(args.ds_name, args.lr))
    dataset_name = 'bill'
    solver = Solver()
    solver.run(dataset_name, args.batch_size, args.epochs, args.lr, args.decay, config,
               trained_model_file)
    """
    python train_recognition.py --batch_size 32 --epochs 100
    """