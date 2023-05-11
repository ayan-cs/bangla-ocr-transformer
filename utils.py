import numpy as np
import os, string, unicodedata, gc, torch, torchvision, time, editdistance
from torch import nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from importlib import import_module

def load_function(attr):
    module_, func = attr.rsplit('.', maxsplit=1)
    return getattr(import_module(module_), func)

def findMaxTextLength(gt_path_train, gt_path_val):
    all_gt_train = os.listdir(gt_path_train)
    all_gt_val = os.listdir(gt_path_val)
    maxlen = 0
    for gt in all_gt_train:
        with open(os.path.join(gt_path_train, gt), 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) > maxlen:
            maxlen = len(text)
    for gt in all_gt_val:
        with open(os.path.join(gt_path_val, gt), 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) > maxlen:
            maxlen = len(text)
    print(maxlen)
    maxlen = (int(maxlen / 100) + 2) * 100
    return maxlen

def generatePlots(train_loss_list, val_loss_list, fig_path):
    if len(train_loss_list) == 0 or len(val_loss_list) == 0:
        print("List empty")
    else:
        min_val_loss = min(val_loss_list)
        epoch = val_loss_list.index(min_val_loss)
        print(f"Optimal point : {epoch+1} epoch with Val loss {min_val_loss}")
        plt.plot(range(len(train_loss_list)), train_loss_list, color='blue', label='Train Loss')
        plt.plot(range(len(val_loss_list)), val_loss_list, color='green', label='Valid loss')
        plt.plot(epoch, min_val_loss, marker = 'v', color = 'red', label = 'Optimal point')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Summary')
        plt.legend()
        plt.savefig(fig_path)