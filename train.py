# -*- coding: utf-8 -*-
# pip install segmentation_models_pytorch
# pip install albumentations
# pip install opencv-python
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold, KFold
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import tifffile as tiff
from PIL import Image
from utils.one_hot import mask2label
from utils.func import SegmentationMetric

import warnings
warnings.filterwarnings("ignore")
# =============================== 初始化 ========================
class Config:
    seed = 2025
    epochs = 20
    batch_size = 2
    n_fold = 2
    learning_rate = 3e-4 
    img_size = 224 
    num_classes = 2 
    print_freq = 2 
    model_save_dir = './weights/'
    result_save_dir = './results/' 

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)   

CFG = Config()

def seed_it(seed):
#     random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True   
    torch.manual_seed(seed)
seed_it(CFG.seed)


# ===============================data aug ========================
def get_train_transforms():
    return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.25),
            A.ShiftScaleRotate(p=0.25),
        ], p=1.)
  
        
def get_val_transforms():
    return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
        ], p=1.)


class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms=None, mode='train'):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.post_transforms = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406, 0], std=[0.229, 0.224, 0.225, 1], max_pixel_value=255.0, p=1.0),
                                ToTensorV2(p=1.0),])
        self.mode = mode
        self.len = len(image_paths)
    def __len__(self):
        return self.len   
    def __getitem__(self, index):
        img_re = tiff.imread(self.image_paths[index]) 
        img_re = img_re[:, :, np.newaxis]
        
        img_rgb = cv2.imread(self.image_paths[index].replace('_MS_RE.TIF', '_D.JPG'))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        w, h = img_re.shape[1], img_re.shape[0]
        img_rgb = cv2.resize(img_rgb, (w, h)) 
        
        # print(img_rgb.shape, img_re.shape)
        img = np.concatenate([img_rgb, img_re], axis=-1)
        mask = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        if self.mode=='train':
            aug_post = self.post_transforms(image=img)
            img = aug_post['image']
            
        mask = mask/255.0
        mask = torch.from_numpy(mask).long()
        
        return img, mask


# =============================== train ========================
def train_model(model, optimizer, lr_scheduler=None, criterion =None, max_epoch=10):
    total_iters = len(train_loader)
    best_miou = 0
    best_epoch = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs = [], [], []
    train_losses = []
    beset_miou = []
    train_iou = []
    train_acc = []
    lrs = []
    train_cpa = []
    min_loss = np.inf
    min_miou = 0
    min_cpa = 0
    min_recall = 0
    best = 0
    decrease = 1
    not_improve = 0
    train_miou = []
    train_recall = []
    train_f1 = []

    # 开始训练
    for epoch in range(max_epoch):
        losses = []
        running_loss = 0
        iou_score = 0
        accuracy = 0
        cpa = 0
        miou1 = 0
        recall = 0
        f1 = 0
        model.train()
        for i, (inputs, labels)  in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) #B,C,H,W
            out = model(inputs)
            loss = criterion(out, labels) 
            
            metric = SegmentationMetric(2) 
            metric.addBatch(out, labels)
            cpa += metric.meanPixelAccuracy()
            miou1 += metric.meanIntersectionOverUnion()
            recall += metric.recall()
            f1 += metric.F1Score()
            accuracy += metric.pixelAccuracy()
            
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient
            lrs.append(optimizer.param_groups[-1]['lr'])
            running_loss += loss.item()

            losses.append(loss.item())
            if CFG.print_freq > 0 and (i % CFG.print_freq == 0):
                print("Epoch:{}/{}..".format(epoch + 1, max_epoch),
                  "Train Loss: {:.2f}..".format(running_loss / len(train_loader)),
                  "Train Acc:{:.2f}..".format(accuracy / len(train_loader)),
                  "train_cpa:{:.2f}..".format(cpa / len(train_loader)),
                  "train_miou:{:.2f}..".format(miou1 / len(train_loader)),
                  "train_recall:{:.2f}..".format(recall / len(train_loader)),
                  "train_f1:{:.2f}..".format(f1 / len(train_loader)),
                     )
        lr_scheduler.step()
        

        val_iou = val_model(model, val_loader)
        train_loss_epochs.append(np.array(losses).mean())
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])

        best_model_path = CFG.model_save_dir + "/" + '_best' + '.pth'
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print("Best epoch/miou: {}/{}".format(best_epoch, best_miou))
            print('per clsIoU:{}'.format('\t'.join(np.stack(val_iou).mean(0).round(3).astype(str)))) 
    return train_loss_epochs, val_mIoU_epochs, lr_epochs


def val_model(model, loader):
    val_iou = []
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            out = out.argmax(1)
            iou = cal_iou(out, labels)
            val_iou.append(iou)
    model.train()
    return val_iou

# 计算IoU
def cal_iou(pred, mask):
    iou_result = []
    for idx in range(CFG.num_classes):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)                  

# =============================== model ========================
from models.vision_transformer import SwinUnet
from config import get_config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml' )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

args = parser.parse_args()
config = get_config(args)

class MyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # self.model = smp.UnetPlusPlus(
        #         encoder_name="resnet34",
        #         encoder_weights="imagenet",
        #         in_channels=4,
        #         decoder_attention_type="scse",
        #         classes=num_classes,
        #         )
        self.model = SwinUnet(config, img_size=CFG.img_size, num_classes=num_classes).cuda()
        self.model.load_from(config)

        
    def forward(self, x):
        #B,C,H,W
        out = self.model(x) #B, .., H,W
        return out 
        


def predict_image_mask_miou(model, image, mask):
    model.eval()
    test_transformer = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406,0], std=[0.229, 0.224, 0.225,1], max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),])
    image = test_transformer(image=image)['image']
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        metric = SegmentationMetric(2)
        metric.addBatch(output, mask)
        score = metric.meanIntersectionOverUnion() 
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
        print("score",score)
    return masked, score

    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # =============================== data  &&  loader========================
    train_image_paths = sorted(glob.glob('./data/*.TIF'))
    train_label_paths = sorted(glob.glob('./data/*.png')) 
    train_image_paths = np.array(train_image_paths)
    train_label_paths = np.array(train_label_paths) 
    
    train_dataset = MyDataset(train_image_paths, train_label_paths, get_train_transforms(), mode='train')
    val_dataset = MyDataset(train_image_paths, train_label_paths, get_val_transforms(), mode='train')
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size*2, shuffle=False, num_workers=4)
    
    # =============================== train========================

    model = MyModel(num_classes=CFG.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1) 
    criterion = nn.CrossEntropyLoss()
    train_loss_epochs, val_mIoU_epochs, lr_epochs = train_model(model, optimizer, scheduler, criterion, max_epoch=CFG.epochs)


    
    # ===============================test & vis ========================
    test_dataset = MyDataset(train_image_paths, train_label_paths, get_val_transforms(), mode='test')
    for i in range(3):
        print("num", i)
        image2, mask2 = test_dataset[i]
        pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)
        mask2 = mask2.numpy()
        pred_mask2 = pred_mask2.numpy()
        mask2 = mask2label(mask2)
        pred_mask2 = mask2label(pred_mask2)
        pred_mask2 = Image.fromarray(np.uint8(pred_mask2))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(image2[:, :, 0:3])
        ax1.set_title('Picture')
        ax1.set_axis_off()
        ax2.imshow(mask2)
        ax2.set_title('Ground truth')
        ax2.set_axis_off()
        ax3.imshow(pred_mask2)
        ax3.set_title('UNet | IoU {:.3f}'.format(score2))
        ax3.set_axis_off()
        plt.savefig('./%s'%CFG.result_save_dir+"%03d"%(i)+'.png')
        plt.show()


        