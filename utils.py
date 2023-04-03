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