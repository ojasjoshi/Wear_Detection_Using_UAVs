## imports!
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import os.path as osp
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
import pickle
import random
from preprocess import process, Imdb
import pdb

# IM_SIZE = (128,128)
IM_SIZE = (512,512)
THRESHOLD = 0.5
IS_CUDA = False

def metric(y_pred, y_true):
    """ absolute error """
    return 1-abs((y_pred.data.cpu().numpy()[0]-y_true.data.cpu().numpy()[0])[0])

def train(train_loader, model, criterion, optimizer, epoch, logger):
    losses = AverageMeter()
    precision = AverageMeter()
    model.train()
    n_batches = len(train_loader)

    for i_batch, (input, target) in enumerate(train_loader):
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        input_var = torch.autograd.Variable(input)

        if(IS_CUDA):
            target = target.cuda(async=True)
            input_var = input_var.cuda()

        y_pred = model.forward(input_var)

        if(IS_CUDA):
            y_pred = y_pred.cuda()

        target_var = torch.autograd.Variable(target)
        # print(y_pred.cpu().data.numpy().shape, target_var.cpu().data.numpy().shape)

        loss = criterion(y_pred, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m = metric(y_pred, target_var)
        precision.update(m, input.size(0))
        losses.update(loss.data[0], input.size(0))
    
    logger.scalar_summary('epoch_loss',losses.avg,epoch)
    print('Epoch: {0}\t, Average_Train_Loss: {1:.2f}, Average_Train_Precision: {1:.2f}'.format(epoch, losses.avg, precision.avg))

def validate(test_loader, model, epoch):
    # log.scalar_summary('tp', 0, 0)
    precision = AverageMeter()
    model.eval()

    for i_batch, (input, target) in enumerate(test_loader):
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        input_var = torch.autograd.Variable(input)
        if(IS_CUDA):
            target = target.cuda(async=True)
            input_var = input_var.cuda()
        
        target_var = torch.autograd.Variable(target)
        y_pred = model.forward(input_var)

        m = metric(y_pred, target_var)
        precision.update(m, input.size(0))

    print('VAL\nEpoch: {0}\t Val Precision: {1:.2f}\nVAL'.format(epoch, precision.avg))
    logger.scalar_summary('val_precision', precision.avg, epoch)


## called per image
def process_test_input(input):
    return torch.from_numpy(np.expand_dims(np.expand_dims(input,axis=0),axis=0))

def test(test_image_data, model):
    precision = AverageMeter()
    model.eval()

    predictions = []
    """ TODO: Check input type requirement """
    test_image_cutouts = [x[0] for x in test_image_data]

    for i_batch, input in enumerate(test_image_cutouts):
        input = process_test_input(input)
        # print("INPUT IMAGE SHAPE:",input.shape)
        input = input.type(torch.FloatTensor)

        input_var = torch.autograd.Variable(input)
        if(IS_CUDA):
            input_var = input_var.cuda()

        y_pred = model.forward(input_var)

        predictions.append(y_pred.data.cpu().numpy()[0][0])

    return predictions


def adjust_learning_rate(optimizer, epoch, decay_rate=0.8, decay_epoch=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Conv2d(in_filters, out_filters, kernel_size=(3,3),stride=(1,1),padding=(1,1)):
    return nn.Conv2d(in_filters,out_filters,kernel_size=kernel_size,stride= stride,padding=padding)

def Max2d(kernel_size=(2,2),stride=2):
    return nn.MaxPool2d(kernel_size=kernel_size,stride=stride)

def get_test_data(data_dir='../test_data/', num_dir=1):
    _ , _ , preprocessed_cutouts, original_img_shape = process(crop_size=IM_SIZE[0], data_dir=data_dir, num_dir=num_dir) # preprocessed cutouts contains list of list of cutouts per image

    return preprocessed_cutouts, original_img_shape


def get_trainval_data(batch_size, train_percent, num_workers=1, data_dir='../data/', num_dir=1):
        
    wear_cut, no_wear_cut, _ , _ = process(crop_size=IM_SIZE[0],data_dir=data_dir,num_dir=num_dir)
    images, labels = Imdb(wear_cut, no_wear_cut)

    train_idx = random.sample(range(0,len(images)),int(len(images)*train_percent))
    
    mask = np.zeros(len(images),dtype=bool)
    mask[train_idx] = True

    train_images = np.asarray(images)[mask]
    train_labels = np.asarray(labels)[mask]
    val_images = np.asarray(images)[~mask]
    val_labels = np.asarray(labels)[~mask]

    train_dataset = DatasetTrainVal(train_images, train_labels, im_size=IM_SIZE) #TODO: initialise
    val_dataset = DatasetTrainVal(val_images, val_labels, im_size=IM_SIZE) #TODO: initialise

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, val_dataloader

class Model(nn.Module):
    def __init__(self, im_size):
        super(Model, self).__init__()
        # im_size = (h,w) 
        self.imsize = im_size
        # self.linearDimension = im_size[0]*im_size[1]*64
        """ WARNING: CHECK THIS SHIT!!! """
        self.linearDimension = int(im_size[0]*im_size[1]*0.25)
        self.features = nn.Sequential(
            Conv2d(1,16),
            Max2d(),
            Conv2d(16,32),
            Max2d(),
            Conv2d(32,64),
            Max2d(),
            Conv2d(64,64),
            Max2d(),
            )
            # size would now be (h/8,w/8,64)
            # input size to score 

        self.score = nn.Sequential(
            nn.Linear(self.linearDimension, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.features(x)
        # pdb.set_trace()
        x = x.view([x.shape[0], self.linearDimension])
        score = self.score(x)
        return score

class DatasetTrainVal(data.Dataset):
    def __init__(self, input_im, target, im_size):
        # input data to this is a list of inputs and targets
        self.input = input_im
        self.target = target
        self.im_size = im_size # (tuple)
        self.length = self.__len__()

    def __getitem__(self, index):
        image = self.input[index]
        level = self.target[index]
        return image, level

    def __len__(self):
        return len(self.input)


class DatasetTest(data.Dataset):
    def __init__(self, vid_data):
        self.vid_data = vid_data
        self.num_classes = 51
        self.length = self.__len__()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: np.array :Feature is a 10 * 512 array
        """
        feature = self.vid_data[index]['features']
        return feature

    def __len__(self):
        return len(self.vid_data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # if type(val) is not float:
        #     print('error')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
