import os
import sys
import cv2
import argparse
import numpy as np
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.samples   = [name for name in os.listdir(args.datapath+'/images') if name[0]!="."]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name   = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'/images/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H,W,C  = image.shape
        mask = cv2.imread(self.args.datapath + '/masks/' + os.path.splitext(name)[0] + '.png', cv2.IMREAD_GRAYSCALE) / 255.0
        pair   = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin

    def __len__(self):
        return len(self.samples)

class Test(object):
    def __init__(self, Data, Model, args):
        ## dataset
        self.args      = args
        self.data      = Data(args)
        self.loader    = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model     = Model(args)
        self.model.train(False)
        self.model.cuda()

    def save_prediction(self):
        print(self.args.datapath.split('/')[-1])
        with torch.no_grad():
            for image, mask, shape, name, origin in self.loader:
                image = image.cuda().float()
                pred = self.model(image, shape)
                pred  = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred>0)] /= (pred>0).float().mean()
                pred[torch.where(pred<0)] /= (pred<0).float().mean()
                pred  = pred.cpu().numpy()*255
                if not os.path.exists(self.args.predpath):
                    os.makedirs(self.args.predpath)
                cv2.imwrite(self.args.predpath+'/'+name[0], np.round(pred))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapaths', default='./data/TestDataset/')
    parser.add_argument('--predpaths', default='./eval/prediction/SIJ/')
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--snapshot', default='./SIJ1/model-128')
    args = parser.parse_args()
    for name in [
        'CVC-300',
        'CVC-ClinicDB',
        'CVC-ColonDB',
        'ETIS-LaribPolypDB',
        'KvasirCapsule']:
        args.datapath = args.datapaths + name
        args.predpath = args.predpaths + name
        t = Test(Data, SANet, args)
        t.save_prediction()