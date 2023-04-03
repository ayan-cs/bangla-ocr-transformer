import numpy as np
import os, unicodedata, torch
from itertools import groupby
from torch import nn
from torchvision.models import resnet50, resnet101
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image

class DataGenerator(Dataset):
    """Generator class with data streaming"""

    def __init__(self, img_source, gt_source, charset, max_text_length, transform):
        #print("Inside init DataGenerator")
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.transform = transform
        self.x = []
        self.y = []
        
        for img in os.listdir(img_source):
            self.x.append(np.array(Image.open(os.path.join(img_source, img))))
            with open(os.path.join(gt_source, img.split('.')[0]+'.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
            self.y.append(text)
        self.size = len(self.y)
        self.x = np.array(self.x)
        self.y = np.array(self.y)


    def __getitem__(self, i):
        img = img = self.x[i]
        
        # making image compatible (3-channel) with resnet
        img = np.repeat(img[..., np.newaxis],3, -1)   
        if self.transform is not None:
            img = self.transform(img)
        y_train = self.tokenizer.encode(self.y[i])
        
        # padding till max length
        #print(self.tokenizer.maxlen, len(y_train), (self.tokenizer.maxlen - len(y_train))<0)
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))
        gt = torch.Tensor(y_train)
        return img, gt

    def __len__(self):
      return self.size

class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""
    def __init__(self, chars, max_text_length):
        self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "PAD", "UNK", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] + chars
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)
        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""
        text = unicodedata.normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")
        text = " ".join(text.split())
        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []
        text = ['SOS'] + list(text) + ['EOS']
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)
        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""
        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")

class LabelSmoothing(nn.Module):
    "Implement label smoothing"
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))