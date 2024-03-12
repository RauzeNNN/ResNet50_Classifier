import torch
from DataLoader import Data_Classifier
from Trainer import Trainer
import random
import os
import numpy as np
from Model import ResNet50Classifier, InceptionV3, DenseNet121
from torch import optim
from torch.utils.data import DataLoader
import argparse
import yaml
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    args = ap.parse_args()
    return args


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return ("{}: {}".format(phase, ", ".join(outputs)))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_input(dataloaders, titles=["Input", 'Target']):
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    train_batch = next(iter(train_loader))
    img, label, _ = train_batch
    print('train image shape:', img.shape)
    print('tarin label shape:', label.shape)

    val_batch = next(iter(val_loader))
    img, label, _ = val_batch
    print('val image shape:', img.shape)
    print('val label shape:', label.shape)

    labels_map = {
        0: 'control',
        1: 'retinal detachment',
    }
    figure = plt.figure(figsize=(24, 12))
    cols, rows = 3, 2
    train_loader = iter(train_loader)
    for i in range(1, cols * rows + 1):
        img, label, img_path = next(train_loader)
        img, label, img_path = img[0], label[0], img_path[0]
        label_idx = torch.nonzero(label)[0].item()
        name = img_path.split('/')[-1]
        figure.add_subplot(rows, cols, i)
        plt.title('{}, {}, {}'.format(
            name, img_path.split('/')[-2], labels_map[label_idx]))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.savefig('batch_sample_train.png')
    plt.clf()

    # figure = plt.figure(figsize=(24, 12))
    # cols, rows = 3, 2
    # val_loader = iter(val_loader)
    # for i in range(1, cols * rows + 1):
    #     img, label, img_path = next(val_loader)
    #     img, label, img_path = img[0], label[0], img_path[0]
    #     label_idx = torch.nonzero(label)[0].item()
    #     name = img_path.split('/')[-1]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title('{}, {}, {}'.format(
    #         name, img_path.split('/')[-2], labels_map[label_idx]))
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.savefig('batch_sample_val.png')
    # plt.clf()


def main(cfg):
    seed = cfg['train_config']['seed']
    seed_everything(seed)
    # model configs
    input_size = (cfg['model_config']['input_size'][0],
                  cfg['model_config']['input_size'][1])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']

    # train configs
    batch_size = cfg['train_config']['batch_size'][0]
    num_workers = cfg['train_config']['num_workers']
    lr_rate = cfg['train_config']['lr_rate'][0]
    Epoch = cfg['train_config']['epochs']
    use_cuda = cfg['train_config']['use_cuda']
    loss_function = cfg['train_config']['loss']
    accuracy_metric = cfg['train_config']['accuracy']
    weight_decay = cfg['train_config']['weight_decay'][0]

    # dataset configs
    train_path = cfg['dataset_config']['train_path']
    val_path = cfg['dataset_config']['val_path']
    aug_rate = cfg['dataset_config']['aug_rate']
    output_save_dir = cfg['dataset_config']['save_dir']

    train_dataset = Data_Classifier(train_path, ch, input_size=input_size, augmentation = cfg['dataset_config']['augmentation'])
    val_dataset = Data_Classifier(val_path, ch, input_size=input_size, augmentation = cfg['dataset_config']['augmentation'])
    print('Train set size:', len(train_dataset))
    print('Val set size:', len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    torch.autograd.set_detect_anomaly(True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
   
    check_input(dataloaders)
    if cfg['model_config']['model']== 'ResNet50':
        model = ResNet50Classifier(ch, num_class, use_cuda)
    elif cfg['model_config']['model']== 'InceptionV3':
        model = InceptionV3(ch, num_class, use_cuda)
    elif cfg['model_config']['model']== 'DenseNet121':
        model = DenseNet121(ch, num_class, use_cuda)
    else:
        print('TODO')

        # model = ResNetScratch(ch, num_class)
    start_epoch = 1
    if cfg['resume']['flag']:
        model.load_state_dict(torch.load(cfg['resume']['path']))
        start_epoch = cfg['resume']['epoch']
    if use_cuda:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        device = "cuda:0"
        dtype = torch.cuda.FloatTensor
        model.to(device=device)
    else:
        device = "cpu"
        model.to(device=device)

    if cfg['train_config']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
             model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    elif cfg['train_config']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid otpimizer "%s"' % cfg['train_config']['optimizer'])

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=30)

    trainer = Trainer(model, dtype, device, output_save_dir, dataloaders, batch_size, optimizer,
                      patience=30, num_epochs=Epoch, loss_function=loss_function, accuracy_metric=accuracy_metric, lr_scheduler=None, start_epoch=start_epoch)
    best_model = trainer.train()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg)
